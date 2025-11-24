"""
Google ADK 图纸参数识别 Agent (ModelScope 版)
使用 Google ADK 框架 + ModelScope API (OpenAI 兼容协议)
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm  # 必须导入这个
from google.adk.tools import FunctionTool
from dotenv import load_dotenv
import mimetypes
import sys
import tempfile
import uuid
import shutil
# 1. 优先加载环境变量 (确保能读到 ADK_LOG_DIR)
load_dotenv()

# 2. 配置日志
# 获取日志目录，优先使用环境变量，否则默认为当前目录下的 logs
log_dir = os.getenv("ADK_LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))

# 处理路径中的 ~ 符号 (例如 ~/tmp/...)
if log_dir.startswith("~"):
    log_dir = os.path.expanduser(log_dir)

handlers = [logging.StreamHandler(sys.stdout)]

try:
    # 尝试创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件 Handler
    log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    handlers.append(file_handler)
    
    # 临时打印一下，确认路径
    print(f"System: 日志将写入 -> {log_file}")
    
except Exception as e:
    print(f"System Warning: 无法创建日志文件 ({e})，将仅输出到控制台")

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers,
    force=True  # 关键：强制覆盖之前的配置（防止被其他库抢先配置）
)

logger = logging.getLogger(__name__)


# ============= 自定义 LiteLlm 类以修复文件格式问题 =============
import hashlib  # <--- 新增导入

# ============= 最终增强版 LiteLlm (MD5 哈希去重版) =============
class ModelScopeLiteLlm(LiteLlm):
    """
    LiteLlm 扩展：
    1. [Fix] 使用 MD5 哈希检测重复文件，彻底防止重复保存。
    2. [Fix] 自动将 PDF 转换为 JPEG 图片再传给 LLM。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 缓存字典：Key 是文件内容的 MD5，Value 是 {'saved_path':..., 'image_path':...}
        self._file_cache = {} 

    @property
    def _temp_upload_dir(self) -> str:
        """动态获取并创建临时目录"""
        temp_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    async def generate_content_async(self, llm_request, stream=False):
        from google.genai import types

        if hasattr(llm_request, 'contents') and llm_request.contents:
            for content in llm_request.contents:
                if not hasattr(content, 'parts') or not content.parts:
                    continue

                new_parts = []
                for part in content.parts:
                    
                    # === 情况 A: 如果是 inline_data (Web 上传) ===
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # 1. 计算内容哈希 (这是唯一可靠的去重标准)
                        file_data = part.inline_data.data
                        file_hash = hashlib.md5(file_data).hexdigest()
                        
                        saved_path = None
                        image_path = None
                        
                        # 2. 检查缓存
                        if file_hash in self._file_cache:
                            # === 命中缓存 ===
                            cache_entry = self._file_cache[file_hash]
                            saved_path = cache_entry['saved_path']
                            image_path = cache_entry.get('image_path')
                            # logger.info(f"检测到重复文件 (Hash: {file_hash[:8]}), 使用缓存路径: {saved_path}")
                        else:
                            # === 首次遇到 ===
                            saved_path = self._save_with_hash_name(part.inline_data, file_hash)
                            
                            # 如果是 PDF，进行转换
                            if part.inline_data.mime_type == "application/pdf":
                                try:
                                    logger.info(f"正在转换 PDF: {saved_path}")
                                    image_path = convert_pdf_to_image(saved_path)
                                except Exception as e:
                                    logger.error(f"PDF 转图片失败: {e}")
                            
                            # 存入缓存
                            self._file_cache[file_hash] = {
                                'saved_path': saved_path,
                                'image_path': image_path
                            }
                            logger.info(f"新文件已保存: {saved_path}")

                        # 3. 构造 Visual Part (给模型看的)
                        visual_part = part
                        
                        # 如果有转换好的图片路径（无论是刚转的还是缓存的），都读出来做成 JPEG Part
                        if image_path and os.path.exists(image_path):
                             try:
                                with open(image_path, "rb") as f:
                                    img_bytes = f.read()
                                visual_part = types.Part(
                                    inline_data=types.Blob(
                                        mime_type="image/jpeg",
                                        data=img_bytes
                                    )
                                )
                             except Exception as e:
                                 logger.error(f"读取缓存图片失败: {e}")

                        # 如果是 PDF 且转换失败了，visual_part 设为 None，避免报错
                        elif part.inline_data.mime_type == "application/pdf" and not image_path:
                            visual_part = None

                        if visual_part:
                            new_parts.append(visual_part)
                        
                        # 4. 注入路径提示
                        path_hint = f"\n[System: File saved at: {saved_path}]\n"
                        new_parts.append(types.Part(text=path_hint))

                    # === 情况 B: 本地文件 ===
                    elif self._extract_file_uri(part):
                        file_uri = self._extract_file_uri(part)
                        inline_part = self._file_to_inline_part(file_uri)
                        if inline_part:
                            new_parts.append(inline_part)
                        new_parts.append(types.Part(text=f"\n[System: File path: {file_uri}]\n"))
                        
                    else:
                        new_parts.append(self._ensure_part_instance(part))

                content.parts = new_parts

        async for response in super().generate_content_async(llm_request, stream=stream):
            yield response

    def _save_with_hash_name(self, inline_data, file_hash) -> str:
        """使用哈希值作为文件名保存，物理层面避免重复"""
        ext = ".jpg"
        if inline_data.mime_type == "image/png":
            ext = ".png"
        elif inline_data.mime_type == "application/pdf":
            ext = ".pdf"
            
        # 文件名里包含 hash，这样即便重启服务，如果 hash 一样也可以考虑复用（虽然这里 self._file_cache 是内存级的）
        file_name = f"upload_{file_hash[:10]}{ext}"
        file_path = os.path.join(self._temp_upload_dir, file_name)
        
        # 只有文件不存在时才写入
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(inline_data.data)
                
        return file_path

    def _ensure_part_instance(self, part):
        from google.genai import types
        if isinstance(part, types.Part):
            return part
        if isinstance(part, dict):
            return types.Part(**part)
        return part

    def _extract_file_uri(self, part) -> Optional[str]:
        file_data = None
        if hasattr(part, 'file_data') and part.file_data:
            file_data = part.file_data
        elif isinstance(part, dict):
            file_data = part.get('file_data') or part.get('fileData')

        if not file_data:
            return None

        if hasattr(file_data, 'file_uri') and file_data.file_uri:
            return file_data.file_uri

        if isinstance(file_data, dict):
            return file_data.get('file_uri') or file_data.get('fileUri')
        return None

    def _file_to_inline_part(self, file_path: str):
        from google.genai import types
        
        resolved_path = file_path
        if file_path.lower().endswith('.pdf'):
            try:
                resolved_path = convert_pdf_to_image(file_path)
            except Exception as e:
                logger.error(f"Local PDF conversion failed: {e}")
                return None

        mime_type, _ = mimetypes.guess_type(resolved_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = "image/jpeg"

        with open(resolved_path, "rb") as f:
            file_bytes = f.read()

        return types.Part(
            inline_data=types.Blob(
                mime_type=mime_type,
                data=file_bytes
            )
        )
# ============= 工具函数 =============

# def read_drawing_file(file_path: str) -> Dict[str, Any]:
#     path = Path(file_path)
#     if not path.exists():
#         return {"error": f"文件不存在: {file_path}"}
#     return {
#         "file_name": path.name,
#         "file_size": path.stat().st_size,
#         "file_type": path.suffix,
#         "exists": True
#     }

# def list_drawing_files(directory: str, pattern: str = "*.jpg") -> List[str]:
#     from glob import glob
#     search_path = os.path.join(directory, pattern)
#     files = glob(search_path)
#     return files
def extract_text_ocr(image_path: str) -> str:
    """
    使用 PaddleOCR (高精度深度学习模型) 从图像中提取文本。
    """
    import os
    import logging
    
    # 1. 基础检查
    if not os.path.exists(image_path):
        return f"错误: 找不到文件 '{image_path}'。"
        
    try:
        # 延迟导入
        from paddleocr import PaddleOCR
    except ImportError:
        return "系统错误: 未安装 paddleocr。请运行: pip install paddlepaddle paddleocr opencv-python-headless"

    try:
        # ------------------------------------------------------------------
        # [修正点 1] 通过 logging 模块屏蔽 Paddle 的调试日志
        # 这样比传参数更稳定，不会报错
        # ------------------------------------------------------------------
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        # ------------------------------------------------------------------
        # 2. 初始化模型
        # ------------------------------------------------------------------
        if not hasattr(extract_text_ocr, "ocr_engine"):
            logging.info("正在初始化 PaddleOCR 模型 (首次运行需下载模型)...")
            
            # [修正点 2] 删除了 show_log=False 参数，防止报错
            # use_angle_cls=True: 支持旋转文字
            # lang="ch": 中英文通用
            extract_text_ocr.ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch")
        
        ocr = extract_text_ocr.ocr_engine
        
        # ------------------------------------------------------------------
        # 3. 执行识别
        # ------------------------------------------------------------------
        # cls=True 启用方向分类器
        result = ocr.ocr(image_path, cls=True)
        
        if not result or result[0] is None:
             return "OCR 完成，但未检测到任何文字。"

        # ------------------------------------------------------------------
        # 4. 格式化输出
        # ------------------------------------------------------------------
        text_lines = []
        for idx in range(len(result)):
            res = result[idx]
            if not res: continue
            for line in res:
                # line 格式: [[x,y], [text, confidence]]
                text_content = line[1][0]
                confidence = line[1][1]
                
                # 过滤低置信度
                if confidence > 0.6:
                    text_lines.append(text_content)
        
        final_text = "\n".join(text_lines)
        
        # 如果太长，截断一下防止撑爆 Context Window
        if len(final_text) > 4000:
            final_text = final_text[:4000] + "\n...(截断)..."
            
        logging.info(f"PaddleOCR 识别成功，提取了 {len(text_lines)} 行文字。")
        return final_text

    except Exception as e:
        logging.error(f"PaddleOCR 运行失败: {e}")
        # 返回明确的错误提示，防止 Agent 盲目重试
        return f"【系统错误】OCR 引擎故障: {str(e)}。请停止使用 OCR 工具，直接尝试通过视觉分析图片。"
    
def validate_parameters(parameters: List[Dict]) -> Dict[str, Any]:
    """
    验证提取出的参数列表是否符合规范，并检查置信度。
    
    在最终输出 JSON 结果之前，应当使用此工具对提取到的参数进行自检。
    
    Args:
        parameters (List[Dict]): 提取到的参数列表，每个字典应包含 name, value, confidence 等字段。
        
    Returns:
        Dict[str, Any]: 验证结果，包含有效参数数量、发现的问题列表 (issues) 以及是否通过验证 (is_valid)。
    """
    valid_count = 0
    issues = []
    for param in parameters:
        if not param.get("name") or not param.get("value"):
            issues.append(f"参数缺少名称或值: {param}")
        # 也可以在这里处理一些数值清洗逻辑
        elif param.get("confidence", 0) < 0.5:
            issues.append(f"低置信度参数: {param['name']}")
        else:
            valid_count += 1
    return {
        "valid_count": valid_count,
        "total_count": len(parameters),
        "issues": issues,
        "is_valid": len(issues) == 0
    }

def convert_pdf_to_image(pdf_path: str, output_dir: Optional[str] = None, dpi: int = 200) -> str:
    """将 PDF 转换为图像"""
    try:
        from pdf2image import convert_from_path
        
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
            
        pdf_name = Path(pdf_path).stem
        output_path = os.path.join(output_dir, f"{pdf_name}_page1.jpg")
        
        # 如果图片已存在，直接返回，避免重复转换节省时间
        if os.path.exists(output_path):
            logger.info(f"使用已存在的缓存图像: {output_path}")
            return output_path

        # 转换 PDF 第一页为图像
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
        if not images:
            raise ValueError("PDF 转换失败")
        
        images[0].save(output_path, 'JPEG')
        logger.info(f"✓ PDF 已转换为图像: {output_path}")
        return output_path
        
    except ImportError:
        logger.error("pdf2image 未安装。请运行: pip install pdf2image")
        logger.error("另外需要安装 poppler: sudo apt-get install poppler-utils")
        raise
    except Exception as e:
        logger.error(f"PDF 转换失败: {e}")
        raise


# ============= Google ADK Agent (ModelScope Modified) =============

class DrawingRecognitionAgent:
    """基于 Google ADK 的图纸识别 Agent (ModelScope Version)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        # 默认使用 ModelScope 上的 Qwen-VL 模型
        # 注意：一定要用 "openai/" 前缀
        model_name: str = "openai/Qwen/Qwen3-VL-235B-A22B-Instruct" 
    ):
        load_dotenv()
        
        # 1. 获取 ModelScope Key
        self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
        
        if not self.api_key:
            raise ValueError("需要配置 MODELSCOPE_API_KEY 环境变量")
        
        self.model_name = model_name
        # 创建工具
        self.tools = self._create_tools()
        
        # 创建 ADK Agent
        self.agent = self._create_agent()
        
        logger.info(f"✓ ADK Agent 初始化完成")
        logger.info(f"  - Provider: ModelScope")
        logger.info(f"  - Model: {model_name}")

    def _load_config(self) -> Dict[str, Any]:
        config_path = Path(__file__).with_name("config.json")
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"读取 config.json 失败: {exc}")
            return {}
        
    def _create_tools(self) -> List:
        return [
            FunctionTool(extract_text_ocr),
            FunctionTool(validate_parameters)
        ]
    
    def _create_agent(self) -> Agent:
        """创建配置了 LiteLLM + ModelScope 的 Agent"""
        
        instruction = """
        你是一个专业的工程图纸分析专家。你的任务是分析图纸并提取关键参数。
        **重要：关于文件路径**
        - 用户上传的文件路径通常会包含在对话上下文中。
        - 当你需要调用 `extract_text_ocr` 时，**必须使用用户提供的真实文件路径**。
        - 如果你不知道文件路径，请不要猜测（不要使用 drawing_image.png），请先询问用户：“请提供图纸文件的路径”。
        **分析步骤：**
        1. 识别图纸类型（机械图/电气图/建筑图等）
        2. 提取图纸标题和编号
        3. 识别比例信息
        4. 提取所有可见的参数（尺寸、规格、材料等）

        **输出要求：**
        - 以JSON格式返回结果
        - 每个参数包含：name（参数名）、value（参数值）、unit（单位）、confidence（置信度0-1）

        **JSON格式示例：**
        ```json
        {
            "drawing_type": "机械工程图",
            "title": "轴承座",
            "drawing_number": "ZC-001",
            "scale": "1:2",
            "parameters": [
                {"name": "外径", "value": "100", "unit": "mm", "confidence": 0.95},
                {"name": "内径", "value": "50", "unit": "mm", "confidence": 0.92}
            ]
        }
        ```
        """
        
        # 2. 使用自定义的 ModelScopeLiteLlm（修复文件格式问题）
        llm_config = ModelScopeLiteLlm(
            model=self.model_name,
            api_key=self.api_key,
            # 强制指向 ModelScope 的 OpenAI 兼容接口
            api_base="https://api-inference.modelscope.cn/v1"
        )
        
        agent = Agent(
            name="drawing_recognition_agent",
            model=llm_config,  # 传入配置好的 LiteLlm 对象
            tools=self.tools,
            instruction=instruction
        )
        
        return agent
