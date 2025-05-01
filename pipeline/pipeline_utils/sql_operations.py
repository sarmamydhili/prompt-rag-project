from typing import List, Dict, Optional
from pipeline.pipeline_utils.db_connections import get_mysql_connection, DBConfig

class SQLOperations:
    def __init__(self):
        self.mysql_conn = get_mysql_connection()

    def get_skills_by_ids(self, skill_ids: List[int]) -> List[Dict]:
        """
        Get skills data from MySQL by skill IDs
        Args:
            skill_ids: List of skill IDs to query
        Returns:
            List of skill data dictionaries
        """
        cursor = self.mysql_conn.cursor()
        skills_data = []

        try:
            query = """
                SELECT
                    s.skill_id,
                    s.skill_name,
                    s.additional_details,
                    s.subject_area,
                    s.subject,
                    t.task_name
                FROM
                    adaptive_skills s
                LEFT JOIN
                    adaptive_task_skills t ON s.skill_id = t.skill_id
                WHERE
                    s.skill_id = %s
            """
            for skill_id in skill_ids:
                cursor.execute(query, (skill_id,))
                result = cursor.fetchone()
                if result:
                    skills_data.append({
                        "skill_id": result[0],
                        "skill_name": result[1],
                        "skill_additional_details": result[2],
                        "subject_area": result[3],
                        "subject": result[4],
                        "task_name": result[5]
                    })
                # Ensure all results are consumed
                while cursor.fetchone() is not None:
                    pass
            return skills_data
        finally:
            cursor.close()

    def get_skills_by_task_name(self, task_name: str) -> List[Dict]:
        """
        Get skills data from MySQL by task name
        Args:
            task_name: Task name to query
        Returns:
            List of skill data dictionaries
        """
        cursor = self.mysql_conn.cursor()
        skills_data = []

        try:
            query = """
                SELECT
                    s.skill_id,
                    s.skill_name,
                    s.additional_details,
                    s.subject_area,
                    s.subject,
                    t.task_name
                FROM
                    adaptive_skills s
                INNER JOIN
                    adaptive_task_skills t ON s.skill_id = t.skill_id
                WHERE
                    t.task_name = %s
            """
            cursor.execute(query, (task_name,))
            results = cursor.fetchall()
            for result in results:
                skills_data.append({
                    "skill_id": result[0],
                    "skill_name": result[1],
                    "skill_additional_details": result[2],
                    "subject_area": result[3],
                    "subject": result[4],
                    "task_name": result[5]
                })
            return skills_data
        finally:
            cursor.close()

    def close(self):
        """Close the MySQL connection"""
        if self.mysql_conn:
            self.mysql_conn.close() 