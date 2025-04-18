import json

class PromptBuilder:
    def __init__(self):
        # Pre-define your templates
        self.system_prompt_template_path = 'pipeline/prompts/generation_system_prompt_template_mc.txt'
        self.user_prompt_template_path = 'pipeline/prompts/generation_usr_prompt_template_mc.txt'

    def load_prompt(self, filepath):
        """
        Load a prompt from a file.
        
        Args:
            filepath (str): Path to the prompt file
            
        Returns:
            str: Content of the prompt file
        """
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except FileNotFoundError as e:
            print(f"Error: Prompt file not found at {filepath}: {e}")
            raise
        except Exception as e:
            print(f"Error reading prompt file at {filepath}: {e}")
            raise

    def create_prompts(self, parameters):
        """
        Create system and user prompts based on parameters.
        
        Args:
            parameters (dict): Dictionary of parameters
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        try:
            # Step 1: Load system and user prompt templates
            system_prompt_template = self.load_prompt(self.system_prompt_template_path)
            user_prompt_template = self.load_prompt(self.user_prompt_template_path)
            
            # Step 2: Prepare skills list nicely formatted
            skills_list_raw = parameters.get('skills_list', [])
            skills_text = "\n".join(f"- {skill}" for skill in skills_list_raw)

            # Step 3: Prepare sample questions section
            sample_questions = parameters.get('sample_questions', [])
            if sample_questions and len(sample_questions) > 0:
                # Format sample questions as a string
                sample_questions_str = "\n".join([
                    f"Question {i+1}: {q.get('question', '')}"
                    for i, q in enumerate(sample_questions)
                ])
                
                # Add the section header and instructions
                sample_questions_section = f"""- Use the provided sample questions as reference to maintain consistency in difficulty level, style, and format.
- Pay attention to the following aspects from the sample questions:
  - Question structure and length
  - Use of mathematical notation
  - Diagram requirements
  - Bloom's Taxonomy levels
  - Difficulty distribution

### Reference Questions:
The following are sample questions that demonstrate the expected difficulty level and style:
{sample_questions_str}"""
            else:
                sample_questions_section = ""

            # Step 4: Fill system prompt
            system_prompt = system_prompt_template.format(
                subject=parameters.get('subject', 'Unknown Subject'),
                subject_id=parameters.get('subject_id', 0),
                subject_area=parameters.get('subject_area', 'Unknown Subject Area'),
                subject_area_id=parameters.get('subject_area_id', 0),
                skill_id=parameters.get('skill_id', 0),
                skill=parameters.get('skill_name', 'Unknown Skill'),
                skill_details=parameters.get('skill_details', 'Unknown Skill Details'),
                skills_list=skills_text,
                num_questions=parameters.get('num_questions', 12),  # 🔥 Number of questions
                sample_questions_section=sample_questions_section
            )

            # Step 5: Fill user prompt
            user_prompt = user_prompt_template.format(
                subject=parameters.get('subject', 'Unknown Subject'),
                subject_id=parameters.get('subject_id', 0),
                subject_area=parameters.get('subject_area', 'Unknown Subject Area'),
                subject_area_id=parameters.get('subject_area_id', 0),
                skill_id=parameters.get('skill_id', 0),
                skill=parameters.get('skill_name', 'Unknown Skill'),
                skill_details=parameters.get('skill_details', 'Unknown Skill Details'),
                skills_list=skills_text,
                num_questions=parameters.get('num_questions', 12),  # 🔥 Number of questions
                sample_questions_section=sample_questions_section
            )

            return system_prompt, user_prompt

        except FileNotFoundError as e:
            print(f"Error: Prompt file not found: {e}")
            return None, None
        except Exception as e:
            print(f"Unexpected exception raised: {e}")
            return None, None
