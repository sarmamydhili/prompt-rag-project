"""Application configuration for Digital SAT generation."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from pipeline.pipeline_utils.db_connections import DBConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "digital_sat_generation")
DEFAULT_CONFIG_PATH = os.path.join(PIPELINE_DIR, "task_config.properties")


@dataclass
class DigitalSatConfig:
    """Runtime configuration loaded from task_config.properties and environment."""

    mongo_server: str = "127.0.0.1"
    mongo_port: str = "27017"
    mongo_db_name: str = "adaptive_learning_docs"
    mongo_questions_collection: str = "dryrun_questions"
    mongo_course_framework_collection: str = "course_framework"
    mongo_output_collection: str = "dryrun_questions"
    mongo_adaptive_db_name: str = "adaptive_learning_docs"
    mysql_host: str = "localhost"
    mysql_database: str = "adaptive_learning"

    llm_model: str = "grok"
    temperature: float = 0.3
    anthropic_llm_model: str = "claude-3-5-sonnet-latest"
    openai_llm_model: str = "gpt-4-0125-preview"
    gemini_llm_model: str = "gemini-1.5-flash"
    deepseek_llm_model: str = "deepseek-reasoner"
    grok_llm_model: str = "grok-3-latest"
    embedding_model: str = "text-embedding-3-small"

    collection_name: str = "digital_sat_rw_questions"
    prompt_version: str = "digital-sat-rw-v1"
    max_retries: int = 2
    similarity_threshold: float = 0.85
    enable_embedding_similarity: bool = False
    validation_mode: str = "draft"
    enable_duplicate_check: bool = False
    task_name: str = "Digital SAT Reading and Writing"
    subject: str = "Reading and Writing"

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    llm_model_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "DigitalSatConfig":
        load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
        path = config_path or DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        parser = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )
        parser.read(path)

        cfg = cls()
        for section in parser.sections():
            for key, value in parser.items(section):
                if hasattr(cfg, key):
                    current = getattr(cfg, key)
                    if isinstance(current, bool):
                        setattr(cfg, key, value.lower() in ("true", "1", "yes"))
                    elif isinstance(current, int):
                        setattr(cfg, key, int(value))
                    elif isinstance(current, float):
                        setattr(cfg, key, float(value))
                    else:
                        setattr(cfg, key, value)

        if "llm_models" in parser:
            section = parser["llm_models"]
            for key in (
                "anthropic_llm_model",
                "openai_llm_model",
                "gemini_llm_model",
                "deepseek_llm_model",
                "grok_llm_model",
                "embedding_model",
            ):
                if key in section:
                    setattr(cfg, key, section[key])
            if "temperature" in section:
                cfg.temperature = float(section["temperature"])

        if "digital_sat" in parser:
            section = parser["digital_sat"]
            if "collection_name" in section:
                cfg.collection_name = section["collection_name"]
            if "prompt_version" in section:
                cfg.prompt_version = section["prompt_version"]
            if "max_retries" in section:
                cfg.max_retries = int(section["max_retries"])
            if "similarity_threshold" in section:
                cfg.similarity_threshold = float(section["similarity_threshold"])
            if "enable_embedding_similarity" in section:
                cfg.enable_embedding_similarity = section[
                    "enable_embedding_similarity"
                ].lower() in ("true", "1", "yes")
            if "validation_mode" in section:
                cfg.validation_mode = section["validation_mode"]
            if "enable_duplicate_check" in section:
                cfg.enable_duplicate_check = section[
                    "enable_duplicate_check"
                ].lower() in ("true", "1", "yes")
            if "task_name" in section:
                cfg.task_name = section["task_name"]
            if "subject" in section:
                cfg.subject = section["subject"]
            if "llm_model" in section:
                cfg.llm_model = section["llm_model"]
            if "openai_llm_model" in section:
                cfg.openai_llm_model = section["openai_llm_model"]
            if "anthropic_llm_model" in section:
                cfg.anthropic_llm_model = section["anthropic_llm_model"]
            if "gemini_llm_model" in section:
                cfg.gemini_llm_model = section["gemini_llm_model"]
            if "deepseek_llm_model" in section:
                cfg.deepseek_llm_model = section["deepseek_llm_model"]
            if "grok_llm_model" in section:
                cfg.grok_llm_model = section["grok_llm_model"]
            if "temperature" in section:
                cfg.temperature = float(section["temperature"])
        elif "retry" in parser and "max_retries" in parser["retry"]:
            cfg.max_retries = int(parser["retry"]["max_retries"])

        cfg.llm_model_params = {
            "anthropic_llm_model": cfg.anthropic_llm_model,
            "openai_llm_model": cfg.openai_llm_model,
            "gemini_llm_model": cfg.gemini_llm_model,
            "deepseek_llm_model": cfg.deepseek_llm_model,
            "grok_llm_model": cfg.grok_llm_model,
            "embedding_model": cfg.embedding_model,
        }

        DBConfig.initialize_from_context(cfg)
        return cfg

    @property
    def active_model_name(self) -> str:
        return str(self.llm_model_params.get(f"{self.llm_model}_llm_model", self.llm_model))
