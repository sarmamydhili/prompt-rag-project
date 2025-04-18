import json
import os

class PromptBuilder:
    def __init__(self):
        # Load prompt templates
        self.system_prompt_template = self._load_template('generation_system_prompt_template_mc.txt')
        self.user_prompt_template = self._load_template('generation_usr_prompt_template_mc.txt')

    def _load_template(self, filename):
        """Load a prompt template from file"""
        try:
            template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', filename)
            with open(template_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading template {filename}: {e}")
            return None

    def _load_sample_questions(self, sample_questions_file):
        """Load sample questions from JSON file"""
        try:
            with open(sample_questions_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sample questions from {sample_questions_file}: {e}")
            return None

    def create_prompts(self, parameters):
        """
        Create system and user prompts based on the given parameters
        Args:
            parameters: Dictionary containing parameters for prompt generation
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            # Step 1: Prepare base parameters
            base_params = {
                'subject': parameters.get('subject', ''),
                'subject_area': parameters.get('subject_area', ''),
                'skill': parameters.get('skill', ''),
                'learning_objectives': parameters.get('learning_objectives', []),
                'num_questions': parameters.get('num_questions', 1)
            }

            # Step 2: Format learning objectives
            learning_objectives = base_params['learning_objectives']
            if learning_objectives:
                learning_objectives_str = "### Learning Objectives:\n" + "\n".join([
                    f"{i+1}. {objective}"
                    for i, objective in enumerate(learning_objectives)
                ])
            else:
                learning_objectives_str = "No specific learning objectives provided."

            # Step 3: Handle sample questions if file name is provided
            sample_questions_section = ""
            sample_questions_file = parameters.get('sample_questions_file')
            print(f"In build_prompt.py, Sample questions file: {sample_questions_file}")    
            if sample_questions_file:
                sample_questions = self._load_sample_questions(sample_questions_file)
                if sample_questions and 'questions' in sample_questions:
                    # Format sample questions as a string
                    sample_questions_str = "\n".join([
                        f"Question {i+1}: {q.get('question', '')}"
                        for i, q in enumerate(sample_questions['questions'])
                    ])
                    sample_questions_section = f"\n### Sample Questions:\n{sample_questions_str}"

            # Step 4: Create final parameters dictionary
            final_params = {
                **base_params,
                'learning_objectives_str': learning_objectives_str,
                'sample_questions_section': sample_questions_section
            }
            #print(f"Final params: {final_params}")
            # Step 5: Generate prompts
            system_prompt = self.system_prompt_template.format(**final_params)
            #print(f"System prompt: {system_prompt}")
            user_prompt = self.user_prompt_template.format(**final_params)
            #print(f"User prompt: {user_prompt}")
            return system_prompt, user_prompt

        except Exception as e:
            print(f"Error creating prompts: {e}")
            return None, None
