-- ACT Math on Standardized Tests package (idempotent)
USE adaptive_learning;

INSERT INTO adaptive_skills (skill_id, skill_name, skill_details, subject_area, Subject, subject_id, subject_area_id, display_skill, additional_details) VALUES (320, 'Number & Quantity', 'Real and Complex Number Systems; Numerical Quantities', 'Math', 'Math', NULL, NULL, 'Number & Quantity', '[]') ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), Subject=VALUES(Subject), display_skill=VALUES(display_skill);
INSERT INTO adaptive_skills (skill_id, skill_name, skill_details, subject_area, Subject, subject_id, subject_area_id, display_skill, additional_details) VALUES (321, 'Algebra', 'Expressions and Equations; Systems of Equations', 'Math', 'Math', NULL, NULL, 'Algebra', '[]') ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), Subject=VALUES(Subject), display_skill=VALUES(display_skill);
INSERT INTO adaptive_skills (skill_id, skill_name, skill_details, subject_area, Subject, subject_id, subject_area_id, display_skill, additional_details) VALUES (322, 'Functions', 'Definition, Notation, and Application; Manipulation and Graph Features', 'Math', 'Math', NULL, NULL, 'Functions', '[]') ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), Subject=VALUES(Subject), display_skill=VALUES(display_skill);
INSERT INTO adaptive_skills (skill_id, skill_name, skill_details, subject_area, Subject, subject_id, subject_area_id, display_skill, additional_details) VALUES (323, 'Geometry', 'Shapes and Solids; Missing Values, Trigonometry, and Conic Sections', 'Math', 'Math', NULL, NULL, 'Geometry', '[]') ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), Subject=VALUES(Subject), display_skill=VALUES(display_skill);
INSERT INTO adaptive_skills (skill_id, skill_name, skill_details, subject_area, Subject, subject_id, subject_area_id, display_skill, additional_details) VALUES (324, 'Statistics & Probability', 'Distributions and Data Collection; Bivariate Data and Probability', 'Math', 'Math', NULL, NULL, 'Statistics & Probability', '[]') ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), Subject=VALUES(Subject), display_skill=VALUES(display_skill);
INSERT INTO adaptive_skills (skill_id, skill_name, skill_details, subject_area, Subject, subject_id, subject_area_id, display_skill, additional_details) VALUES (325, 'Integrating Essential Skills', 'Core Concepts in Complex Problems; Nonroutine Multi-step Problems', 'Math', 'Math', NULL, NULL, 'Integrating Essential Skills', '[]') ON DUPLICATE KEY UPDATE skill_name=VALUES(skill_name), skill_details=VALUES(skill_details), subject_area=VALUES(subject_area), Subject=VALUES(Subject), display_skill=VALUES(display_skill);

INSERT INTO adaptive_tasks (adaptive_task_name, adaptive_task_description, task_designation)
SELECT 'ACT Math', 'ACT Math MCQ practice', 'general'
WHERE NOT EXISTS (
  SELECT 1 FROM adaptive_tasks WHERE adaptive_task_name = 'ACT Math'
);

SET @act_task_id = (SELECT adaptive_task_id FROM adaptive_tasks
  WHERE adaptive_task_name = 'ACT Math' LIMIT 1);

INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) VALUES (@act_task_id, 'ACT Math', 320);
INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) VALUES (@act_task_id, 'ACT Math', 321);
INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) VALUES (@act_task_id, 'ACT Math', 322);
INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) VALUES (@act_task_id, 'ACT Math', 323);
INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) VALUES (@act_task_id, 'ACT Math', 324);
INSERT IGNORE INTO adaptive_task_skills (task_id, task_name, skill_id) VALUES (@act_task_id, 'ACT Math', 325);

INSERT IGNORE INTO adaptive_package_tasks (adaptive_package_id, adaptive_task_id)
VALUES (3, @act_task_id);

SELECT adaptive_package_id, adaptive_package_name FROM adaptive_packages WHERE adaptive_package_id=3;
SELECT adaptive_task_id, adaptive_task_name FROM adaptive_tasks WHERE adaptive_task_name='ACT Math';
SELECT skill_id, skill_name FROM adaptive_skills WHERE skill_id BETWEEN 320 AND 325 ORDER BY skill_id;

