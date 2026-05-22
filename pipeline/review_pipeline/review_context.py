import os
import configparser
from typing import List, Optional


class ReviewContext:
    """Load settings from review_config.properties only (not task_config.properties)."""

    def __init__(self, config_path: Optional[str] = None):
        pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or os.path.join(pipeline_dir, "review_config.properties")
        self.project_root = os.path.dirname(os.path.dirname(pipeline_dir))

        parser = configparser.ConfigParser()
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Review config not found: {self.config_path}")
        parser.read(self.config_path)

        mongo = parser["mongodb"]
        self.mongo_server = mongo.get("mongo_server", "127.0.0.1")
        self.mongo_port = mongo.get("mongo_port", "27017")
        self.mongo_db_name = mongo.get("mongo_db_name", "adaptive_learning_docs")
        self.mongo_questions_collection = mongo.get("mongo_questions_collection", "dryrun_questions")

        review = parser["review"]
        self.subject = review.get("subject", "").strip()
        self.skill = review.get("skill", "").strip() or None
        level_min = review.get("level_num_min", "").strip()
        self.level_num_min = int(level_min) if level_min else None
        limit = review.get("limit", "").strip()
        self.limit = int(limit) if limit else None

        models = parser["models"]
        providers = models.get("providers", "grok,anthropic")
        self.providers = [p.strip() for p in providers.split(",") if p.strip()]

        llm = parser["llm"]
        self.temperature = float(llm.get("temperature", "0.0"))
        self.llm_model_params = {
            "openai_llm_model": llm.get("openai_llm_model"),
            "anthropic_llm_model": llm.get("anthropic_llm_model"),
            "gemini_llm_model": llm.get("gemini_llm_model"),
            "deepseek_llm_model": llm.get("deepseek_llm_model"),
            "grok_llm_model": llm.get("grok_llm_model"),
            "temperature": self.temperature,
        }

        output = parser["output"]
        report_dir = output.get("report_dir", "pipeline/review_reports")
        if not os.path.isabs(report_dir):
            report_dir = os.path.join(self.project_root, report_dir)
        self.report_dir = report_dir

    def resolve_report_path(self, filename: str) -> str:
        return os.path.join(self.report_dir, filename)
