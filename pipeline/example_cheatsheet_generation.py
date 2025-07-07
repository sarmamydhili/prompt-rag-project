#!/usr/bin/env python3
"""
Example script demonstrating how to use the CheatSheetGenerator workflow
"""

import json
from pipeline.generate_cheatsheets import CheatSheetGenerator


def example_basic_usage():
    """Example of basic usage with default settings"""
    
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
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
    
    # Run workflow
    result = generator.run_workflow(
        subject="AP Physics",
        topic="Kinematics",
        provider="openai",
        model="gpt-4o"
    )
    
    print_result(result)


def example_custom_settings():
    """Example with custom settings"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Settings")
    print("=" * 60)
    
    config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    generator = CheatSheetGenerator(config)
    
    # Run workflow with custom settings
    result = generator.run_workflow(
        subject="AP Calculus BC",
        topic="Limits and Continuity",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        save_to_mongo=True,
        save_to_file=True
    )
    
    print_result(result)


def example_step_by_step():
    """Example showing step-by-step execution"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Step-by-Step Execution")
    print("=" * 60)
    
    config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    generator = CheatSheetGenerator(config)
    
    # Step 1: Read course framework
    print("🔍 Step 1: Reading course framework...")
    step_1_result = generator.step_1_read_course_framework("AP Chemistry", "Atomic Structure")
    
    if not step_1_result["success"]:
        print(f"❌ Step 1 failed: {step_1_result['error']}")
        return
    
    print(f"✅ Step 1 completed: {step_1_result['message']}")
    
    # Step 2: Build prompt
    print("\n📝 Step 2: Building prompt...")
    step_2_result = generator.step_2_build_prompt(step_1_result, "Atomic Structure")
    
    if not step_2_result["success"]:
        print(f"❌ Step 2 failed: {step_2_result['error']}")
        return
    
    print(f"✅ Step 2 completed: {step_2_result['message']}")
    
    # Step 3: Call LLM API
    print("\n🤖 Step 3: Calling LLM API...")
    step_3_result = generator.step_3_call_llm_api(step_2_result, "openai", "gpt-4o")
    
    if not step_3_result["success"]:
        print(f"❌ Step 3 failed: {step_3_result['error']}")
        return
    
    print(f"✅ Step 3 completed: {step_3_result['message']}")
    
    # Step 4: Save response
    print("\n💾 Step 4: Saving response...")
    step_4_result = generator.step_4_save_response(step_3_result, save_to_mongo=True, save_to_file=True)
    
    if not step_4_result["success"]:
        print(f"❌ Step 4 failed: {step_4_result['error']}")
        return
    
    print(f"✅ Step 4 completed: {step_4_result['message']}")
    
    # Print final result
    print("\n🎉 All steps completed successfully!")
    if step_4_result["mongo_saved"]:
        print(f"   MongoDB ID: {step_4_result['mongo_id']}")
    if step_4_result["file_saved"]:
        print(f"   File saved: {step_4_result['file_path']}")


def example_all_topics():
    """Example showing processing all topics for a subject"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Processing All Topics")
    print("=" * 60)
    
    config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    generator = CheatSheetGenerator(config)
    
    # Process all topics for a subject
    result = generator.run_workflow(
        subject="AP Physics",
        topic="*",  # Special value to process all topics
        provider="openai",
        model="gpt-4o",
        save_to_mongo=False,
        save_to_file=True
    )
    
    print_all_topics_result(result)


def example_error_handling():
    """Example showing error handling"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Error Handling")
    print("=" * 60)
    
    config = {
        "openai_llm_model": "gpt-4o",
        "anthropic_llm_model": "claude-3-5-sonnet-20241022",
        "gemini_llm_model": "gemini-1.5-pro",
        "deepseek_llm_model": "deepseek-chat",
        "grok_llm_model": "grok-3"
    }
    
    generator = CheatSheetGenerator(config)
    
    # Try with non-existent subject/topic
    result = generator.run_workflow(
        subject="Non-Existent Subject",
        topic="Some Topic",
        provider="openai"
    )
    
    print_result(result)


def print_result(result):
    """Helper function to print workflow results"""
    
    if result["workflow_success"]:
        print("\n🎉 Workflow completed successfully!")
        
        # Print save information
        save_info = result["final_result"]["save_info"]
        if save_info["mongo_saved"]:
            print(f"   📊 MongoDB ID: {save_info['mongo_id']}")
        if save_info["file_saved"]:
            print(f"   📁 File saved: {save_info['file_path']}")
        
        # Print a preview of the cheat sheet
        cheat_sheet = result["final_result"]["cheat_sheet"]
        if isinstance(cheat_sheet, dict):
            print(f"\n📋 Cheat Sheet Preview:")
            print(f"   Subject: {cheat_sheet.get('subject', 'N/A')}")
            print(f"   Topic: {cheat_sheet.get('topic', 'N/A')}")
            print(f"   Sections: {len(cheat_sheet.get('sections', []))}")
            
            # Show first section if available
            sections = cheat_sheet.get('sections', [])
            if sections:
                first_section = sections[0]
                print(f"   First Section: {first_section.get('section_title', 'N/A')}")
                print(f"   Key Points: {len(first_section.get('key_points', []))}")
                print(f"   Formulas: {len(first_section.get('formulas', []))}")
        
    else:
        print(f"\n❌ Workflow failed: {result['error']}")
        
        # Print step-by-step errors if available
        if "steps" in result:
            for step_name, step_result in result["steps"].items():
                if not step_result["success"]:
                    print(f"   {step_name}: {step_result['error']}")


def print_all_topics_result(result):
    """Helper function to print batch processing results"""
    
    if result["workflow_success"]:
        print("\n🎉 Batch processing completed successfully!")
        
        summary = result["summary"]
        print(f"\n📊 Summary:")
        print(f"   Total topics: {summary['total_processed']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']}")
        
        # Show details for successful topics
        successful_topics = []
        failed_topics = []
        
        for topic, topic_result in result["topic_results"].items():
            if topic_result["workflow_success"]:
                successful_topics.append(topic)
            else:
                failed_topics.append(topic)
        
        if successful_topics:
            print(f"\n✅ Successful topics ({len(successful_topics)}):")
            for topic in successful_topics:
                print(f"   - {topic}")
        
        if failed_topics:
            print(f"\n❌ Failed topics ({len(failed_topics)}):")
            for topic in failed_topics:
                error = result["topic_results"][topic].get("error", "Unknown error")
                print(f"   - {topic}: {error}")
        
    else:
        print(f"\n❌ Batch processing failed: {result['error']}")


def main():
    """Main function to run all examples"""
    
    print("🚀 CheatSheetGenerator Examples")
    print("This script demonstrates various ways to use the CheatSheetGenerator workflow.")
    print("Make sure you have:")
    print("1. MongoDB running with course framework data")
    print("2. LLM API keys configured in environment variables")
    print("3. Required dependencies installed")
    print("\n" + "=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_settings()
        example_step_by_step()
        example_all_topics()
        example_error_handling()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 