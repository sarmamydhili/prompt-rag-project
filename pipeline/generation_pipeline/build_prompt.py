import json
import os

class PromptBuilder:
    def __init__(self, system_prompt_template_path, user_prompt_template_path):
        # Load prompt templates
        print(f"Loading prompt templates from {system_prompt_template_path} and {user_prompt_template_path}")   
        self.system_prompt_template = self._load_template(system_prompt_template_path)
        self.user_prompt_template = self._load_template(user_prompt_template_path)

    def _load_template(self, filename):
        """Load a prompt template from file"""
        print(f"in _load_template Loading template from {filename}")
        try:
            # Ensure filename does not include a path
            filename = os.path.basename(filename)
            template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts', filename)
            with open(template_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading template {filename}: {e}")
            return None

    #def _load_sample_questions(self, sample_questions_file):
    #    """Load and format sample questions from JSON file"""
    #    try:
    #        with open(sample_questions_file, 'r') as f:
    #            sample_questions = json.load(f)
    #        
    #        if sample_questions and 'questions' in sample_questions:
    #            # Format sample questions as a string
    #            sample_questions_str = "\n".join([
    #                f"Question {i+1}: {q.get('question', '')}"
    #                for i, q in enumerate(sample_questions['questions'])
    #            ])
    #            return f"\n### Sample Questions:\n{sample_questions_str}"
    #        return ""
    #    except Exception as e:
    #        print(f"Error loading sample questions from {sample_questions_file}: {e}")
    #        return ""

    def _format_learning_objectives(self, learning_objectives):
        """Format learning objectives into a string"""
        if learning_objectives:
            print(f"Learning objectives: {learning_objectives}")
            return "### Learning Objectives:\n" + "\n".join([
                f"{i+1}. {objective}"
                for i, objective in enumerate(learning_objectives)
            ])
        return "No specific learning objectives provided."

    def create_prompts(self, parameters):
        """
        Create system and user prompts for question generation
        Args:
            parameters: Dictionary containing parameters for prompt generation
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            # Step 1: Prepare base parameters
            base_params = {
                'subject': parameters.get('subject', ''),
                'subject_id': parameters.get('subject_id'),
                'subject_area': parameters.get('subject_area', ''),
                'subject_area_id': parameters.get('subject_area_id'),
                'question': parameters.get('question', ''),
                'skill': parameters.get('skill', ''),
                'task_name': parameters.get('task_name', ''),
                'learning_objectives': parameters.get('learning_objectives', []),
                'num_questions': parameters.get('num_questions', 1),
                'sample_questions_section': parameters.get('sample_questions_section', '')
            }
            print('I am here',1)
            print('learning_objectives',base_params['learning_objectives'])
            # Step 2: Format learning objectives
            learning_objectives_str = self._format_learning_objectives(base_params['learning_objectives'])
            print('I am here',2) 
            # Step 3: Create final parameters dictionary
            final_params = {
                **base_params,
                'learning_objectives': learning_objectives_str
            }
            print('I am here',3)
            #print(f"Final params: {final_params}")
            # Step 4: Generate prompts
            system_prompt = self.system_prompt_template.format(**final_params)
            print('I am here',4)
            print(f"System prompt: {system_prompt}")
            user_prompt = self.user_prompt_template.format(**final_params)
            print('I am here',5)
            print(f"User prompt: {user_prompt}")
            return system_prompt, user_prompt

        except Exception as e:
            print(f"Error creating prompts: {e}")
            return None, None

    def create_enhance_prompts(self, parameters):
        """
        Create system and user prompts for question enhancement
        Args:
            parameters: Dictionary containing parameters for prompt enhancement
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            # Step 1: Prepare base parameters
            base_params = {
                'subject': parameters.get('subject', ''),
                'subject_id': parameters.get('subject_id'),
                'subject_area': parameters.get('subject_area', ''),
                'subject_area_id': parameters.get('subject_area_id'),
                'question': parameters.get('question', ''),
                'skill': parameters.get('skill', ''),
                'skill_name': parameters.get('skill_name', ''),
                'skill_id': parameters.get('skill_id'),
                'multiple_choices': parameters.get('multiple_choices', []),
                'correct_answer': parameters.get('correct_answer', ''),
                'level': parameters.get('level', ''),
                'level_num': parameters.get('level_num', 0),
                'requires_diagram': parameters.get('requires_diagram', False)
            }

            # Step 2: Generate prompts
            system_prompt = self.system_prompt_template.format(**base_params)
            user_prompt = self.user_prompt_template.format(**base_params)
            return system_prompt, user_prompt

        except Exception as e:
            print(f"Error creating enhancement prompts: {e}")
            return None, None
