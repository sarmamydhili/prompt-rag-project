# Cheat Sheet Generator

A modular Python workflow for generating AP-level cheat sheets using MongoDB course frameworks, LLM APIs, and comprehensive prompt engineering.

## Overview

The CheatSheetGenerator provides a complete workflow for creating educational cheat sheets that serve as quick reference guides for AP exam preparation. The system follows a 4-step modular approach:

1. **Read Course Framework** - Extract learning objectives from MongoDB
2. **Build Prompts** - Create system and user prompts with objectives
3. **Call LLM API** - Generate cheat sheet content using various LLM providers
4. **Save Response** - Store results in MongoDB and/or files

## Features

- **Modular Design**: Each step returns results that feed into the next step
- **Multiple LLM Providers**: Support for OpenAI, Anthropic, Gemini, DeepSeek, and Grok
- **Flexible Storage**: Save to MongoDB collection (`adaptive_concepts`) and/or JSON files
- **Comprehensive Prompts**: AP-level prompts with LaTeX formatting support
- **Error Handling**: Robust error handling at each step
- **Subject Support**: Physics, Chemistry, Mathematics, and more

## Prerequisites

1. **MongoDB**: Running with course framework data
2. **Environment Variables**: LLM API keys configured
3. **Dependencies**: Required Python packages installed

### Environment Variables

Set up your `.env` file with the following API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
XAI_API_KEY=your_grok_api_key
```

### MongoDB Setup

Ensure your MongoDB contains:
- `course_framework` collection with subject frameworks
- `adaptive_concepts` collection for storing generated cheat sheets

## Usage

### Basic Usage

```python
from pipeline.generate_cheatsheets import CheatSheetGenerator

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

# Run complete workflow
result = generator.run_workflow(
    subject="AP Physics",
    topic="Kinematics",
    provider="openai",
    model="gpt-4o",
    save_to_mongo=True,
    save_to_file=True
)

if result["workflow_success"]:
    print("✅ Cheat sheet generated successfully!")
    print(f"MongoDB ID: {result['final_result']['save_info']['mongo_id']}")
    print(f"File: {result['final_result']['save_info']['file_path']}")
else:
    print(f"❌ Failed: {result['error']}")
```

### Step-by-Step Usage

```python
# Step 1: Read course framework
step_1_result = generator.step_1_read_course_framework("AP Physics", "Kinematics")

# Step 2: Build prompts
step_2_result = generator.step_2_build_prompt(step_1_result)

# Step 3: Call LLM API
step_3_result = generator.step_3_call_llm_api(step_2_result, "openai", "gpt-4o")

# Step 4: Save response
step_4_result = generator.step_4_save_response(step_3_result, save_to_mongo=True, save_to_file=True)
```

### Different LLM Providers

```python
# OpenAI
result = generator.run_workflow(
    subject="AP Calculus BC",
    topic="Limits and Continuity",
    provider="openai",
    model="gpt-4o"
)

# Anthropic Claude
result = generator.run_workflow(
    subject="AP Chemistry",
    topic="Atomic Structure",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Google Gemini
result = generator.run_workflow(
    subject="AP Biology",
    topic="Cell Biology",
    provider="gemini",
    model="gemini-1.5-pro"
)

### Processing All Topics

To process all topics for a subject, use `topic="*"`:

```python
# Process all topics for AP Physics
result = generator.run_workflow(
    subject="AP Physics",
    topic="*",  # Special value to process all topics
    provider="openai",
    model="gpt-4o",
    save_to_mongo=True,
    save_to_file=True
)

if result["workflow_success"]:
    print(f"✅ Batch processing completed!")
    print(f"Summary: {result['summary']}")
    print(f"Successful topics: {result['successful_topics']}/{result['total_topics']}")
```

## Output Structure

### MongoDB Document

```json
{
  "_id": "ObjectId(...)",
  "content_type": "cheat_sheet",
  "raw_response": "...",
  "json_response": {
    "cheat_sheet": {
      "model_name": "gpt-4o",
      "subject": "AP Physics",
      "topic": "Kinematics",
      "content_type": "cheat_sheet",
      "difficulty_level": "AP Exam Level",
      "sections": [
        {
          "section_title": "Core Concepts",
          "content": "Section content with LaTeX formatting...",
          "key_points": ["Point 1", "Point 2", "Point 3"],
          "formulas": [
            {
              "name": "Velocity Formula",
              "formula": "\\[ v = \\frac{dx}{dt} \\]",
              "description": "Instantaneous velocity is the derivative of position"
            }
          ]
        }
      ],
      "learning_objectives": ["objective1", "objective2"]
    }
  },
  "response_type": "json",
  "provider": "openai",
  "model": "gpt-4o",
  "created_at": "2024-01-01T12:00:00",
  "metadata": {
    "subject": "AP Physics",
    "topic": "Kinematics",
    "objectives_count": 5
  }
}
```

### File Output

Generated files are saved in `generated_cheatsheets/` directory with the format:
`cheatsheet_{subject}_{topic}_{timestamp}.json`

## Configuration

### LLM Models

Supported models by provider:

- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- **Gemini**: `gemini-1.5-pro`, `gemini-1.5-flash`
- **DeepSeek**: `deepseek-chat`, `deepseek-coder`
- **Grok**: `grok-3`

### Prompt Templates

The system uses two prompt templates:
- `pipeline/prompts/concept_prompts/cheatsheet_system_prompt.txt`
- `pipeline/prompts/concept_prompts/cheatsheet_user_prompt.txt`

These templates support:
- LaTeX mathematical notation
- Subject-specific formatting (Physics, Chemistry, Mathematics)
- AP exam level content
- Comprehensive cheat sheet structure

## Error Handling

The system provides comprehensive error handling:

```python
result = generator.run_workflow(...)

if not result["workflow_success"]:
    # Check individual step errors
    for step_name, step_result in result["steps"].items():
        if not step_result["success"]:
            print(f"{step_name}: {step_result['error']}")
```

Common error scenarios:
- **MongoDB Connection Issues**: Check MongoDB service and connection settings
- **Missing Course Framework**: Ensure subject/unit exists in database
- **LLM API Errors**: Verify API keys and model availability
- **Prompt Loading Errors**: Check prompt file paths

## Examples

Run the example script to see various usage patterns:

```bash
python pipeline/example_cheatsheet_generation.py
```

This script demonstrates:
- Basic usage
- Custom settings
- Step-by-step execution
- Error handling

## File Structure

```
pipeline/
├── generate_cheatsheets.py          # Main generator class
├── example_cheatsheet_generation.py # Usage examples
├── README_cheatsheet_generator.md   # This documentation
├── prompts/
│   └── concept_prompts/
│       ├── cheatsheet_system_prompt.txt
│       └── cheatsheet_user_prompt.txt
└── pipeline_utils/
    ├── mongo_operations.py          # MongoDB operations
    ├── llm_connections.py           # LLM API connections
    └── db_connections.py            # Database configuration
```

## Contributing

To extend the system:

1. **Add New LLM Provider**: Extend `LLMConnections` class
2. **Modify Prompts**: Update prompt templates in `concept_prompts/`
3. **Add Storage Options**: Extend `step_4_save_response` method
4. **Support New Subjects**: Update subject-specific requirements in prompts

## Troubleshooting

### Common Issues

1. **"No course framework found"**
   - Verify subject exists in MongoDB `course_framework` collection
   - Check subject name spelling and case

2. **"LLM API returned empty response"**
   - Verify API key is valid
   - Check model name is correct
   - Ensure sufficient API credits

3. **"Failed to save to MongoDB"**
   - Check MongoDB connection
   - Verify `adaptive_concepts` collection exists
   - Check write permissions

4. **"Prompt file not found"**
   - Verify prompt files exist in correct location
   - Check file permissions

### Debug Mode

Enable debug output by modifying the print statements in the generator class or by adding logging configuration.

## License

This project is part of the prompt_rag_project and follows the same licensing terms. 