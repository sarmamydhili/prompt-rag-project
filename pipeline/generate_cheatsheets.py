import json
import os
import configparser
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Import pipeline utilities
from pipeline.pipeline_utils.mongo_operations import MongoOperations
from pipeline.pipeline_utils.llm_connections import LLMConnections
from pipeline.pipeline_utils.db_connections import DBConfig


class CheatSheetGenerator:
    """
    Modular workflow for generating AP-level cheat sheets
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cheat sheet generator with configuration
        
        Args:
            config: Configuration dictionary containing LLM and database settings
        """
        self.config = config
        
        # Initialize database configuration
        self._initialize_db_config()
        
        self.mongo_ops = MongoOperations()
        self.llm_connections = LLMConnections(config)
        
        # Initialize output directory
        self.output_dir = Path("generated_cheatsheets")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load prompts
        self.system_prompt = self._load_system_prompt()
        self.user_prompt = self._load_user_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from file"""
        prompt_path = Path("pipeline/prompts/concept_prompts/cheatsheet_system_prompt.txt")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt not found at {prompt_path}")
    
    def _load_user_prompt(self) -> str:
        """Load the user prompt from file"""
        prompt_path = Path("pipeline/prompts/concept_prompts/cheatsheet_user_prompt.txt")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"User prompt not found at {prompt_path}")
    
    def _initialize_db_config(self):
        """Initialize database configuration from task_config.properties"""
        try:
            # Load task_config.properties
            config_path = Path("pipeline/task_config.properties")
            if not config_path.exists():
                raise FileNotFoundError(f"task_config.properties not found at {config_path}")
            
            config_parser = configparser.ConfigParser()
            config_parser.read(config_path)
            
            # Create a simple context object with database settings
            class Context:
                pass
            
            context = Context()
            
            # MongoDB settings
            context.mongo_server = config_parser.get('mongodb', 'mongo_server', fallback='127.0.0.1')
            context.mongo_port = config_parser.get('mongodb', 'mongo_port', fallback='27017')
            context.mongo_db_name = config_parser.get('mongodb', 'mongo_db_name', fallback='prompt_project')
            context.mongo_questions_collection = config_parser.get('mongodb', 'mongo_questions_collection', fallback='questions')
            context.mongo_course_framework_collection = config_parser.get('mongodb', 'mongo_course_framework_collection', fallback='course_framework')
            context.mongo_output_collection = config_parser.get('mongodb', 'mongo_output_collection', fallback='output_questions_enhanced')
            context.mongo_adaptive_db_name = config_parser.get('mongodb', 'mongo_adaptive_db_name', fallback='adaptive_learning_docs')
            
            # MySQL settings
            context.mysql_host = config_parser.get('mysql', 'mysql_host', fallback='localhost')
            context.mysql_database = config_parser.get('mysql', 'mysql_database', fallback='adaptive_learning')
            
            # Chroma settings
            context.chroma_collection_name = config_parser.get('chroma', 'chroma_collection_name', fallback='questions_collection')
            context.chroma_persist_directory = config_parser.get('chroma', 'chroma_persist_directory', fallback='chroma_db')
            
            # Initialize DBConfig with context
            DBConfig.initialize_from_context(context)
            
            print(f"✅ Database configuration initialized successfully")
            print(f"   MongoDB: {DBConfig.MONGO_SERVER}:{DBConfig.MONGO_PORT}/{DBConfig.MONGO_DB_NAME}")
            
        except Exception as e:
            print(f"❌ Error initializing database configuration: {str(e)}")
            raise
    
    def step_1_read_course_framework(self, subject: str, topic: str) -> Dict[str, Any]:
        """
        Step 1: Read course framework from MongoDB and get objectives
        
        Args:
            subject: The subject name (e.g., "AP Physics")
            topic: The topic name (e.g., "Kinematics") - same as unit
            
        Returns:
            Dictionary containing step result with objectives and metadata
        """
        print(f"🔍 Step 1: Reading course framework for {subject} - {topic}")
        
        try:
            # Get course framework
            framework = self.mongo_ops.get_course_framework_by_subject(subject)
            if not framework:
                raise ValueError(f"No course framework found for subject: {subject}")
            
            # Get unit objectives (unit is same as topic)
            objectives = self.mongo_ops.get_unit_objectives(subject, topic)
            if not objectives:
                raise ValueError(f"No objectives found for topic: {topic}")
            
            print(f"✅ Found {len(objectives)} objectives for {topic}")
            
            result = {
                "success": True,
                "subject": subject,
                "topic": topic,
                "objectives": objectives,
                "framework": framework,
                "message": f"Successfully retrieved {len(objectives)} objectives"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in Step 1: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to read course framework: {str(e)}"
            }
    
    def step_2_build_prompt(self, step_1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Build prompt using objectives and subject
        
        Args:
            step_1_result: Result from step 1 containing objectives and topic
            
        Returns:
            Dictionary containing built prompts
        """
        topic = step_1_result["topic"]
        print(f"📝 Step 2: Building prompts for topic: {topic}")
        
        try:
            if not step_1_result["success"]:
                raise ValueError("Step 1 failed, cannot build prompt")
            
            subject = step_1_result["subject"]
            objectives = step_1_result["objectives"]
            
            # Format learning objectives
            learning_objectives_text = "\n".join([f"- {obj}" for obj in objectives])
            
            # Build system prompt (skill is same as topic)
            system_prompt = self.system_prompt.format(
                subject=subject,
                skill=topic,
                learning_objectives_section=f"\n### Learning Objectives:\n{learning_objectives_text}" if objectives else ""
            )
            
            # Build user prompt (skill is same as topic)
            user_prompt = self.user_prompt.format(
                subject=subject,
                skill=topic,
                learning_objectives_section=learning_objectives_text if objectives else "No specific learning objectives provided."
            )
            
            print(f"✅ Built prompts successfully")
            print(f"   System prompt length: {len(system_prompt)}")
            print(f"   User prompt length: {len(user_prompt)}")
            
            result = {
                "success": True,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "subject": subject,
                "topic": topic,
                "objectives_count": len(objectives),
                "message": "Successfully built prompts"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in Step 2: {str(e)}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to build prompt: {str(e)}"
            }
    
    def step_3_call_llm_api(self, step_2_result: Dict[str, Any], provider: str = "openai", model: Optional[str] = None) -> Dict[str, Any]:
        """
        Step 3: Call ONLY the specified LLM API with built prompts
        """
        print(f"🤖 Step 3: Calling {provider} API")
        
        try:
            if not step_2_result["success"]:
                raise ValueError("Step 2 failed, cannot call LLM API")
            
            system_prompt = step_2_result["system_prompt"]
            user_prompt = step_2_result["user_prompt"]
            
            # Only call the specified provider
            if provider not in ["openai", "anthropic", "gemini", "deepseek", "grok"]:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            response = self.llm_connections.call_llm_api(
                provider=provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=0.1
            )
            
            if not response:
                raise ValueError("LLM API returned empty response")
            
            # Try to parse JSON response
            try:
                json_response = json.loads(response)
                response_type = "json"
            except json.JSONDecodeError:
                json_response = None
                response_type = "text"
            
            print(f"✅ LLM API call successful")
            
            result = {
                "success": True,
                "raw_response": response,
                "json_response": json_response,
                "response_type": response_type,
                "provider": provider,
                "model": model,
                "subject": step_2_result.get("subject"),
                "topic": step_2_result.get("topic"),
                "objectives_count": step_2_result.get("objectives_count"),
                "message": "Successfully called LLM API"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in Step 3: {str(e)}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to call LLM API: {str(e)}"
            }
    
    def step_4_save_response(self, step_3_result: Dict[str, Any], save_to_mongo: bool = True, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Step 4: Save response to file and/or MongoDB
        
        Args:
            step_3_result: Result from step 3 containing LLM response
            save_to_mongo: Whether to save to MongoDB
            save_to_file: Whether to save to file
            
        Returns:
            Dictionary containing save results
        """
        print(f"💾 Step 4: Saving response")
        
        try:
            if not step_3_result["success"]:
                raise ValueError("Step 3 failed, cannot save response")
            
            save_results = {
                "success": True,
                "mongo_saved": False,
                "file_saved": False,
                "mongo_id": None,
                "file_path": None,
                "message": "Save operation completed"
            }
            
            # Save to MongoDB if requested
            if save_to_mongo:
                mongo_result = self._save_to_mongo(step_3_result)
                save_results["mongo_saved"] = mongo_result["success"]
                save_results["mongo_id"] = mongo_result.get("mongo_id")
                if not mongo_result["success"]:
                    print(f"⚠️  MongoDB save failed: {mongo_result['error']}")
            
            # Save to file if requested
            if save_to_file:
                file_result = self._save_to_file(step_3_result)
                save_results["file_saved"] = file_result["success"]
                save_results["file_path"] = file_result.get("file_path")
                if not file_result["success"]:
                    print(f"⚠️  File save failed: {file_result['error']}")
            
            # Check if at least one save operation succeeded
            if not save_results["mongo_saved"] and not save_results["file_saved"]:
                save_results["success"] = False
                save_results["message"] = "All save operations failed"
            
            print(f"✅ Save operation completed")
            return save_results
            
        except Exception as e:
            print(f"❌ Error in Step 4: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to save response: {str(e)}"
            }
    
    def _save_to_mongo(self, step_3_result: Dict[str, Any]) -> Dict[str, Any]:
        """Save response to MongoDB adaptive_concepts collection"""
        try:
            # Get the clean cheat sheet content (same logic as file saving)
            raw_response = step_3_result.get("raw_response", "")
            json_response = step_3_result.get("json_response")
            
            # If we have a JSON response, use it directly
            if json_response:
                cheat_sheet_content = json_response
            else:
                # Clean the raw response - remove markdown code blocks if present
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove "```json"
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove "```"
                cleaned_response = cleaned_response.strip()
                
                try:
                    cheat_sheet_content = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    # If still can't parse, use as is
                    cheat_sheet_content = {"raw_content": cleaned_response}
            
            # Extract the inner cheat sheet content if it's wrapped
            if isinstance(cheat_sheet_content, dict) and "cheat_sheet" in cheat_sheet_content:
                cheat_sheet_content = cheat_sheet_content["cheat_sheet"]
            
            # Prepare document for MongoDB - store the cheat sheet content as-is
            document = {
                "content_type": "cheat_sheet",
                "cheat_sheet": cheat_sheet_content,  # Store the clean content directly
                "provider": step_3_result["provider"],
                "model": step_3_result["model"],
                "created_at": datetime.now(),
                "metadata": {
                    "subject": step_3_result.get("subject"),
                    "topic": step_3_result.get("topic"),
                    "objectives_count": step_3_result.get("objectives_count")
                }
            }
            
            # Save to adaptive_concepts collection
            result = self.mongo_ops.mongo_db['adaptive_concepts'].insert_one(document)
            
            return {
                "success": True,
                "mongo_id": str(result.inserted_id),
                "message": "Successfully saved to MongoDB"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to save to MongoDB: {str(e)}"
            }
    
    def _save_to_file(self, step_3_result: Dict[str, Any]) -> Dict[str, Any]:
        """Save only the pretty-formatted JSON response (cheat sheet) to file"""
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            subject = step_3_result.get("subject", "unknown")
            topic = step_3_result.get("topic", "unknown")
            filename = f"cheatsheet_{subject}_{topic}_{timestamp}.json"
            file_path = self.output_dir / filename
            
            # Get the JSON response and clean it
            raw_response = step_3_result.get("raw_response", "")
            json_response = step_3_result.get("json_response")
            
            # If we have a JSON response, use it directly
            if json_response:
                cheat_sheet_json = json_response
            else:
                # Clean the raw response - remove markdown code blocks if present
                cleaned_response = raw_response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]  # Remove "```json"
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]  # Remove "```"
                cleaned_response = cleaned_response.strip()
                
                try:
                    cheat_sheet_json = json.loads(cleaned_response)
                except json.JSONDecodeError:
                    # If still can't parse, use as is
                    cheat_sheet_json = {"raw_content": cleaned_response}
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cheat_sheet_json, f, indent=2, ensure_ascii=False, default=str)
            
            return {
                "success": True,
                "file_path": str(file_path),
                "message": "Successfully saved to file"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to save to file: {str(e)}"
            }
    
    def get_all_topics_for_subject(self, subject: str) -> List[str]:
        """
        Get all topics (units) for a given subject from the course framework
        
        Args:
            subject: The subject name (e.g., "AP Physics")
            
        Returns:
            List of topic names
        """
        try:
            framework = self.mongo_ops.get_course_framework_by_subject(subject)
            if not framework:
                return []
            
            topics = []
            for unit in framework.get('units', []):
                topics.append(unit.get('unit', ''))
            
            return [topic for topic in topics if topic]  # Filter out empty topics
            
        except Exception as e:
            print(f"Error getting topics for subject {subject}: {str(e)}")
            return []
    
    def run_workflow(self, subject: str, topic: str, provider: str = "openai", 
                    model: Optional[str] = None, save_to_mongo: bool = True, 
                    save_to_file: bool = True) -> Dict[str, Any]:
        """
        Run the complete workflow for generating a cheat sheet
        
        Args:
            subject: The subject name (e.g., "AP Physics")
            topic: The topic name (e.g., "Kinematics") - same as unit
            provider: LLM provider (openai, anthropic, gemini, deepseek)
            model: Specific model to use (optional)
            save_to_mongo: Whether to save to MongoDB
            save_to_file: Whether to save to file
            
        Returns:
            Dictionary containing results from all steps
        """
        print(f"🚀 Starting cheat sheet generation workflow")
        print(f"   Subject: {subject}")
        print(f"   Topic: {topic}")
        print(f"   Provider: {provider}")
        print(f"   Model: {model or 'default'}")
        print("=" * 60)
        
        # Handle '*' case - process all topics
        if topic == '*':
            return self._run_workflow_for_all_topics(subject, provider, model, save_to_mongo, save_to_file)
        
        workflow_result = {
            "workflow_success": False,
            "steps": {},
            "final_result": None,
            "error": None
        }
        
        try:
            # Step 1: Read course framework
            step_1_result = self.step_1_read_course_framework(subject, topic)
            workflow_result["steps"]["step_1"] = step_1_result
            
            if not step_1_result["success"]:
                workflow_result["error"] = step_1_result["error"]
                return workflow_result
            
            # Step 2: Build prompt
            step_2_result = self.step_2_build_prompt(step_1_result)
            workflow_result["steps"]["step_2"] = step_2_result
            
            if not step_2_result["success"]:
                workflow_result["error"] = step_2_result["error"]
                return workflow_result
            
            # Step 3: Call LLM API
            step_3_result = self.step_3_call_llm_api(step_2_result, provider, model)
            workflow_result["steps"]["step_3"] = step_3_result
            
            if not step_3_result["success"]:
                workflow_result["error"] = step_3_result["error"]
                return workflow_result
            
            # Step 4: Save response
            step_4_result = self.step_4_save_response(step_3_result, save_to_mongo, save_to_file)
            workflow_result["steps"]["step_4"] = step_4_result
            
            # Set final result
            workflow_result["workflow_success"] = step_4_result["success"]
            workflow_result["final_result"] = {
                "cheat_sheet": step_3_result["json_response"] if step_3_result["json_response"] else step_3_result["raw_response"],
                "save_info": step_4_result
            }
            
            print("=" * 60)
            if workflow_result["workflow_success"]:
                print(f"✅ Workflow completed successfully!")
                if step_4_result["mongo_saved"]:
                    print(f"   MongoDB ID: {step_4_result['mongo_id']}")
                if step_4_result["file_saved"]:
                    print(f"   File saved: {step_4_result['file_path']}")
            else:
                print(f"❌ Workflow failed: {workflow_result['error']}")
            
            return workflow_result
            
        except Exception as e:
            workflow_result["error"] = str(e)
            print(f"❌ Workflow error: {str(e)}")
            return workflow_result
        
        finally:
            # Don't close connections here as they might be reused
            pass
    
    def _run_workflow_for_all_topics(self, subject: str, provider: str, 
                                   model: Optional[str], save_to_mongo: bool, 
                                   save_to_file: bool) -> Dict[str, Any]:
        """
        Run workflow for all topics in a subject when topic='*'
        
        Args:
            subject: The subject name
            provider: LLM provider
            model: Specific model to use
            save_to_mongo: Whether to save to MongoDB
            save_to_file: Whether to save to file
            
        Returns:
            Dictionary containing results from all topics
        """
        print(f"🌟 Processing ALL topics for subject: {subject}")
        
        # Get all topics for the subject
        topics = self.get_all_topics_for_subject(subject)
        
        if not topics:
            return {
                "workflow_success": False,
                "error": f"No topics found for subject: {subject}",
                "message": "Failed to get topics from course framework"
            }
        
        print(f"📋 Found {len(topics)} topics: {', '.join(topics)}")
        
        all_results = {
            "workflow_success": True,
            "subject": subject,
            "total_topics": len(topics),
            "successful_topics": 0,
            "failed_topics": 0,
            "topic_results": {},
            "summary": {}
        }
        
        for i, topic in enumerate(topics, 1):
            print(f"\n{'='*60}")
            print(f"📝 Processing topic {i}/{len(topics)}: {topic}")
            print(f"{'='*60}")
            
            try:
                # Run workflow for this topic
                topic_result = self.run_workflow(
                    subject=subject,
                    topic=topic,
                    provider=provider,
                    model=model,
                    save_to_mongo=save_to_mongo,
                    save_to_file=save_to_file
                )
                
                all_results["topic_results"][topic] = topic_result
                
                if topic_result["workflow_success"]:
                    all_results["successful_topics"] += 1
                    print(f"✅ Topic '{topic}' completed successfully")
                else:
                    all_results["failed_topics"] += 1
                    print(f"❌ Topic '{topic}' failed: {topic_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                all_results["failed_topics"] += 1
                all_results["topic_results"][topic] = {
                    "workflow_success": False,
                    "error": str(e),
                    "message": f"Exception during processing: {str(e)}"
                }
                print(f"❌ Topic '{topic}' failed with exception: {str(e)}")
        
        # Create summary
        all_results["summary"] = {
            "total_processed": len(topics),
            "successful": all_results["successful_topics"],
            "failed": all_results["failed_topics"],
            "success_rate": f"{(all_results['successful_topics'] / len(topics)) * 100:.1f}%"
        }
        
        # Set overall success based on whether any topics succeeded
        all_results["workflow_success"] = all_results["successful_topics"] > 0
        
        print(f"\n{'='*60}")
        print(f"🎯 BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"📊 Summary for {subject}:")
        print(f"   Total topics: {all_results['total_topics']}")
        print(f"   Successful: {all_results['successful_topics']}")
        print(f"   Failed: {all_results['failed_topics']}")
        print(f"   Success rate: {all_results['summary']['success_rate']}")
        
        if all_results["successful_topics"] > 0:
            print(f"✅ Batch processing completed with {all_results['successful_topics']} successful topics")
        else:
            print(f"❌ Batch processing failed - no topics were processed successfully")
        
        return all_results
    
    def close_connections(self):
        """Close database connections when done"""
        if hasattr(self, 'mongo_ops'):
            self.mongo_ops.close()


def main():
    """Main function to demonstrate usage"""
    
    # Configuration
    config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    # Initialize generator
    generator = CheatSheetGenerator(config)
    
    # Example usage - single topic
    result = generator.run_workflow(
        subject="AP Statistics",
        topic="*",
        provider="openai",
        model="gpt-4o",
        save_to_mongo=True,
        save_to_file=False
    )
    
    if result["workflow_success"]:
        print("\n🎉 Cheat sheet generated successfully!")
        #print(f"Final result: {result['final_result']}")
    else:
        print(f"\n❌ Workflow failed: {result['error']}")
    
if __name__ == "__main__":
    main() 