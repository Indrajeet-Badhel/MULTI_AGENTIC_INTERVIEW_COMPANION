from flask import Flask, request, jsonify, send_file, session
import json
import os
import random
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText  # FIXED: Capital letters
from email.mime.multipart import MIMEMultipart  # FIXED: Capital letters
from email.mime.base import MIMEBase  # FIXED: Capital letters
from email import encoders
from fpdf import FPDF
import re
import threading
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'advanced-ai-interview-2025')

class AdvancedInterviewSystem:
    def __init__(self):
        self.question_categories = {
            'technical': ['frontend', 'backend', 'database', 'system_design', 'security'],
            'coding': ['algorithms', 'data_structures', 'problem_solving', 'optimization'],
            'behavioral': ['teamwork', 'leadership', 'communication', 'problem_solving'],
            'industry': ['trends', 'best_practices', 'tools', 'methodologies']
        }
        
        self.difficulty_levels = {
            'easy': {'complexity': 1, 'keywords': ['basic', 'simple', 'fundamental']},
            'medium': {'complexity': 2, 'keywords': ['intermediate', 'practical', 'real-world']},
            'hard': {'complexity': 3, 'keywords': ['advanced', 'complex', 'architectural']}
        }
        
        self.trending_topics = self.fetch_trending_topics()
        
    def fetch_trending_topics(self):
        """Fetch trending tech topics from free APIs"""
        try:
            # Using free GitHub API to get trending repositories
            response = requests.get('https://api.github.com/search/repositories?q=created:>2024-01-01&sort=stars&order=desc&per_page=20', timeout=10)
            if response.status_code == 200:
                repos = response.json()['items']
                topics = []
                for repo in repos[:10]:
                    if repo.get('topics'):
                        topics.extend(repo['topics'])
                return list(set(topics))[:20]
        except:
            pass
        
        # Fallback trending topics for 2025
        return ['ai', 'machine-learning', 'react', 'nextjs', 'typescript', 'python', 
                'kubernetes', 'docker', 'microservices', 'cloud-native', 'serverless',
                'graphql', 'web3', 'blockchain', 'cybersecurity', 'devops']

    def generate_question_with_ollama(self, category, difficulty, position, experience, trending_context=""):
        """Generate questions using Ollama with enhanced prompts"""
        try:
            prompt = self.create_enhanced_prompt(category, difficulty, position, experience, trending_context)
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 300,
                        "stop": ["Question:", "Q:", "\n\n"]
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                return self.clean_question(result)
                
        except Exception as e:
            print(f"Ollama error: {e}")
            
        return self.get_fallback_question(category, difficulty, position)

    def generate_question_with_huggingface(self, category, difficulty, position, experience):
        """Generate questions using free Hugging Face models"""
        try:
            # Using the free question generation model
            api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            
            context = f"Generate a {difficulty} {category} interview question for {position} ({experience} level):"
            
            payload = {
                "inputs": context,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.8,
                    "do_sample": True
                }
            }
            
            response = requests.post(api_url, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    return self.clean_question(generated_text)
                    
        except Exception as e:
            print(f"Hugging Face error: {e}")
            
        return self.get_fallback_question(category, difficulty, position)

    def create_enhanced_prompt(self, category, difficulty, position, experience, trending_context):
        """Create sophisticated prompts for question generation"""
        base_prompt = f"""
        You are an expert technical interviewer for a {position} position.
        
        Candidate Profile:
        - Position: {position}
        - Experience: {experience}
        - Interview Category: {category}
        - Difficulty Level: {difficulty}
        
        Current Industry Context:
        - Trending Technologies: {', '.join(self.trending_topics[:5])}
        - Year: 2025
        {trending_context}
        
        Generate ONE specific, practical interview question that:
        1. Tests real-world problem-solving skills
        2. Is appropriate for {experience} level candidates
        3. Relates to current industry practices and trends
        4. Requires detailed, thoughtful answers
        5. Focuses on {category} concepts
        
        The question should be challenging but fair, and encourage the candidate to demonstrate both theoretical knowledge and practical experience.
        
        Question:"""
        
        return base_prompt

    def clean_question(self, raw_question):
        """Clean and validate generated questions"""
        if not raw_question:
            return None
            
        # Remove common prefixes and clean up
        cleaned = raw_question.strip()
        prefixes = ['Question:', 'Q:', 'Here\'s a question:', 'Generate a question based on']
        
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                
        # Extract the main question
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 20 and ('?' in line or 
                line.lower().startswith(('what', 'how', 'why', 'describe', 'explain', 'design', 'implement'))):
                return line
                
        # Return cleaned text if it looks like a valid question
        if len(cleaned) > 20 and len(cleaned) < 500:
            return cleaned
            
        return None

    def get_fallback_question(self, category, difficulty, position):
        """Enhanced fallback questions with 2025 context"""
        questions_db = {
            'technical': {
                'easy': [
                    "What are the key differences between REST and GraphQL APIs in modern web development?",
                    "How do you implement responsive design using CSS Grid and Flexbox in 2025?",
                    "Explain the concept of component-based architecture in modern frontend frameworks.",
                    "What are the benefits of using TypeScript over JavaScript in large-scale applications?",
                    "How do you handle state management in React applications using modern hooks?"
                ],
                'medium': [
                    "Design a caching strategy for a high-traffic web application using Redis and CDN.",
                    "How would you implement real-time features in a web application using WebSockets and Server-Sent Events?",
                    "Explain your approach to implementing micro-frontends in a large enterprise application.",
                    "How do you optimize database queries for a system handling millions of records?",
                    "Design a CI/CD pipeline for a microservices architecture using modern DevOps tools."
                ],
                'hard': [
                    "Architect a globally distributed system that can handle 100 million concurrent users with 99.99% uptime.",
                    "How would you implement a real-time collaborative editing system like Google Docs?",
                    "Design a fault-tolerant event-driven architecture for a financial trading platform.",
                    "Implement a custom load balancer that can handle traffic spikes and automatic failover.",
                    "How would you design a system for processing and analyzing real-time streaming data at petabyte scale?"
                ]
            },
            'coding': {
                'easy': [
                    "Write a function to find the most frequent element in an array with O(n) time complexity.",
                    "Implement a simple LRU cache using Python dictionaries and demonstrate its usage.",
                    "Create a function that validates if a string contains balanced parentheses, brackets, and braces.",
                    "Write an algorithm to merge two sorted arrays without using extra space.",
                    "Implement a function to detect if a linked list has a cycle and return the starting node."
                ],
                'medium': [
                    "Design and implement a thread-safe singleton pattern in Python with lazy initialization.",
                    "Write an algorithm to find the shortest path in a weighted graph using Dijkstra's algorithm.",
                    "Implement a trie data structure for autocomplete functionality with prefix matching.",
                    "Create a function to serialize and deserialize a binary tree to/from a string representation.",
                    "Write an algorithm to find all anagrams of a string in a given list of words."
                ],
                'hard': [
                    "Implement a distributed consistent hashing algorithm for load balancing across multiple servers.",
                    "Design a lock-free data structure for a high-concurrency environment.",
                    "Write an algorithm to solve the traveling salesman problem using dynamic programming with bitmasks.",
                    "Implement a real-time collaborative text editor with operational transformation.",
                    "Create a system for rate limiting API requests using the sliding window log algorithm."
                ]
            },
            'behavioral': {
                'easy': [
                    "Tell me about a time when you had to learn a new technology quickly for a project.",
                    "Describe a situation where you had to work with a difficult team member.",
                    "How do you stay updated with the latest technology trends and developments?",
                    "Tell me about a project you're particularly proud of and why.",
                    "Describe a time when you had to explain a complex technical concept to a non-technical person."
                ],
                'medium': [
                    "Tell me about a time when you had to make a critical technical decision under pressure.",
                    "Describe a situation where you had to refactor legacy code. What was your approach?",
                    "How do you handle conflicting priorities when working on multiple projects?",
                    "Tell me about a time when you had to advocate for a technical solution that others disagreed with.",
                    "Describe your experience mentoring junior developers and the challenges you faced."
                ],
                'hard': [
                    "Tell me about a time when you had to lead a technical team through a major system failure.",
                    "Describe a situation where you had to make architectural decisions that would impact the entire organization.",
                    "How do you balance technical debt with feature development in a fast-paced environment?",
                    "Tell me about a time when you had to implement a solution that required significant organizational change.",
                    "Describe your approach to building and scaling engineering teams while maintaining code quality."
                ]
            }
        }
        
        category_questions = questions_db.get(category, questions_db['technical'])
        difficulty_questions = category_questions.get(difficulty, category_questions['medium'])
        
        return random.choice(difficulty_questions)

    def score_answer_with_ai(self, question, answer, category, difficulty):
        """Score answers using AI with detailed feedback"""
        if not answer or answer.upper() == 'SKIPPED':
            return {
                'score': 0,
                'feedback': 'Question was skipped - no answer provided',
                'strengths': [],
                'improvements': ['Provide an answer to demonstrate knowledge']
            }
        
        # Try Ollama for scoring
        try:
            scoring_prompt = f"""
            You are an expert technical interviewer evaluating a candidate's response.
            
            Question: {question}
            Answer: {answer}
            Category: {category}
            Difficulty: {difficulty}
            
            Evaluate this answer on a scale of 0-10 considering:
            1. Technical accuracy and correctness
            2. Completeness and depth of explanation
            3. Practical understanding and real-world application
            4. Communication clarity and structure
            5. Use of relevant examples or experiences
            
            Be strict in your evaluation. Average answers should score 4-6, good answers 7-8, excellent answers 9-10.
            
            Provide your evaluation in this exact format:
            Score: [0-10]
            Feedback: [2-3 sentences explaining the score]
            Strengths: [List 2-3 positive aspects]
            Improvements: [List 2-3 areas for improvement]
            """
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "llama3.1",
                    "prompt": scoring_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 400}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                return self.parse_ai_evaluation(result)
                
        except Exception as e:
            print(f"AI scoring error: {e}")
        
        # Fallback to rule-based scoring
        return self.rule_based_scoring(answer, question, category, difficulty)

    def parse_ai_evaluation(self, evaluation_text):
        """Parse AI evaluation response"""
        try:
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', evaluation_text, re.IGNORECASE)
            feedback_match = re.search(r'Feedback:\s*(.+?)(?=Strengths:|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            strengths_match = re.search(r'Strengths:\s*(.+?)(?=Improvements:|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            improvements_match = re.search(r'Improvements:\s*(.+)', evaluation_text, re.IGNORECASE | re.DOTALL)
            
            score = float(score_match.group(1)) if score_match else 3.0
            score = min(10, max(0, score))
            
            feedback = feedback_match.group(1).strip() if feedback_match else "AI evaluation provided"
            strengths = self.parse_list_items(strengths_match.group(1)) if strengths_match else []
            improvements = self.parse_list_items(improvements_match.group(1)) if improvements_match else []
            
            return {
                'score': round(score, 1),
                'feedback': feedback[:300],
                'strengths': strengths[:3],
                'improvements': improvements[:3]
            }
            
        except Exception as e:
            print(f"Error parsing AI evaluation: {e}")
            return self.rule_based_scoring("", "", "", "")

    def parse_list_items(self, text):
        """Parse list items from text"""
        if not text:
            return []
        
        items = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                # Remove bullet points and numbering
                line = re.sub(r'^[-â€¢*\d+\.)\s]+', '', line).strip()
                if line:
                    items.append(line)
        
        return items[:3]

    def rule_based_scoring(self, answer, question, category, difficulty):
        """Enhanced rule-based scoring system with stricter criteria"""
        if not answer or len(answer.strip()) < 10:
            return {
                'score': 0,
                'feedback': 'Answer too short or empty',
                'strengths': [],
                'improvements': ['Provide a more detailed answer', 'Include specific examples']
            }
        
        score = 2.0  # Lower base score for stricter evaluation
        strengths = []
        improvements = []
        
        # Length analysis (stricter requirements)
        length = len(answer)
        if length > 400:
            score += 2.5
            strengths.append("Comprehensive and detailed response")
        elif length > 250:
            score += 2.0
            strengths.append("Good level of detail provided")
        elif length > 150:
            score += 1.0
            strengths.append("Adequate explanation given")
        else:
            improvements.append("Provide more detailed explanations with examples")
        
        # Technical keyword analysis
        answer_lower = answer.lower()
        
        technical_keywords = {
            'technical': ['api', 'database', 'framework', 'architecture', 'scalability', 'performance', 'security', 'design', 'implementation'],
            'coding': ['algorithm', 'complexity', 'optimization', 'data structure', 'function', 'class', 'method', 'time complexity', 'space complexity'],
            'behavioral': ['team', 'project', 'challenge', 'solution', 'leadership', 'communication', 'collaboration', 'experience', 'result']
        }
        
        relevant_keywords = technical_keywords.get(category, technical_keywords['technical'])
        keyword_count = sum(1 for keyword in relevant_keywords if keyword in answer_lower)
        
        if keyword_count >= 5:
            score += 2.0
            strengths.append("Uses relevant technical terminology effectively")
        elif keyword_count >= 3:
            score += 1.5
            strengths.append("Demonstrates good technical vocabulary")
        elif keyword_count >= 1:
            score += 0.5
            strengths.append("Shows some technical understanding")
        else:
            improvements.append("Include more relevant technical terms and concepts")
        
        # Quality indicators (stricter requirements)
        quality_indicators = ['example', 'experience', 'because', 'however', 'therefore', 'specifically', 'implementation', 'approach', 'solution']
        quality_count = sum(1 for indicator in quality_indicators if indicator in answer_lower)
        
        if quality_count >= 4:
            score += 2.0
            strengths.append("Provides clear reasoning and specific examples")
        elif quality_count >= 2:
            score += 1.0
            strengths.append("Shows some analytical thinking")
        else:
            improvements.append("Include specific examples and detailed reasoning")
        
        # Difficulty adjustment (stricter for higher difficulties)
        difficulty_adjustments = {'easy': 1.0, 'medium': 0.9, 'hard': 0.8}
        score *= difficulty_adjustments.get(difficulty, 1.0)
        
        # Cap score at 10 and ensure minimum standards
        score = min(10, max(1, score))
        
        # Generate feedback based on stricter criteria
        if score >= 8:
            feedback = "Excellent answer demonstrating strong knowledge and communication skills."
        elif score >= 6:
            feedback = "Good answer with solid understanding, but could be enhanced with more depth."
        elif score >= 4:
            feedback = "Average answer showing basic understanding, needs more detail and examples."
        elif score >= 2:
            feedback = "Below average answer, requires significant improvement in content and depth."
        else:
            feedback = "Poor answer, needs substantial improvement in all areas."
        
        return {
            'score': round(score, 1),
            'feedback': feedback,
            'strengths': strengths[:3],
            'improvements': improvements[:3]
        }

# Initialize the system
interview_system = AdvancedInterviewSystem()

@app.route('/')
def index():
    return open("index.html", encoding="utf-8").read()

@app.route('/get_question', methods=['POST'])
def get_question():
    try:
        data = request.json
        question_index = data.get('questionIndex', 0)
        candidate_info = data.get('candidateInfo', {})
        
        if 'interview_session' not in session:
            session['interview_session'] = {
                'questions': [],
                'responses': {},
                'start_time': datetime.now().isoformat(),
                'candidate_info': candidate_info
            }
        
        # Check if interview is complete
        total_questions = int(candidate_info.get('questionCount', 5))
        if question_index >= total_questions:
            return jsonify({"finished": True})
        
        # Determine question category and difficulty
        categories = ['technical', 'coding', 'behavioral']
        if question_index < len(categories):
            category = categories[question_index]
        else:
            category = random.choice(categories)
        
        difficulty = candidate_info.get('difficulty', 'medium')
        if difficulty == 'mixed':
            difficulty = random.choice(['easy', 'medium', 'hard'])
        
        position = candidate_info.get('position', 'Software Developer')
        experience = candidate_info.get('experience', 'Junior')
        
        # Generate question using multiple methods
        question = None
        
        # Try Ollama first
        question = interview_system.generate_question_with_ollama(
            category, difficulty, position, experience
        )
        
        # Fallback to Hugging Face
        if not question:
            question = interview_system.generate_question_with_huggingface(
                category, difficulty, position, experience
            )
        
        # Final fallback
        if not question:
            question = interview_system.get_fallback_question(category, difficulty, position)
        
        # Store question in session
        session['interview_session']['questions'].append({
            'index': question_index,
            'question': question,
            'category': category,
            'difficulty': difficulty,
            'timestamp': datetime.now().isoformat()
        })
        session.modified = True
        
        return jsonify({
            "question": question,
            "question_type": category.title(),
            "difficulty": difficulty,
            "question_number": question_index + 1,
            "total_questions": total_questions
        })
        
    except Exception as e:
        print(f"Error in get_question: {e}")
        return jsonify({"error": "Failed to generate question"}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    try:
        data = request.json
        question_index = data.get('questionIndex', 0)
        question = data.get('question', '')
        answer = data.get('answer', '')
        question_type = data.get('questionType', '').lower()
        
        if 'interview_session' not in session:
            return jsonify({"error": "No active interview session"}), 400
        
        # Store response
        session['interview_session']['responses'][f'q_{question_index}'] = {
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'timestamp': datetime.now().isoformat()
        }
        session.modified = True
        
        return jsonify({"status": "answer_recorded"})
        
    except Exception as e:
        print(f"Error in submit_answer: {e}")
        return jsonify({"error": "Failed to submit answer"}), 500

@app.route('/finish_interview', methods=['POST'])
def finish_interview():
    try:
        if 'interview_session' not in session:
            return jsonify({"error": "No active interview session"}), 400
        
        interview_data = session['interview_session']
        interview_data['end_time'] = datetime.now().isoformat()
        
        # Score the interview
        scores = score_complete_interview(interview_data)
        
        # Generate and save report
        candidate_name = interview_data['candidate_info'].get('name', 'Anonymous')
        position = interview_data['candidate_info'].get('position', 'Software Developer')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"interview_report_{candidate_name.replace(' ', '_')}_{timestamp}.pdf"
        
        generate_detailed_report(interview_data, scores, report_filename)
        
        # Send email if configured
        email_sent = False
        candidate_email = interview_data['candidate_info'].get('email')
        if candidate_email and os.getenv('EMAIL_FROM') and os.getenv('EMAIL_PASS'):
            email_sent = send_interview_report_email(candidate_email, candidate_name, report_filename)
        
        # Store report filename in session for download
        session['last_report'] = report_filename
        session.modified = True
        
        return jsonify({
            "status": "completed",
            "scores": scores,
            "report_generated": True,
            "email_sent": email_sent
        })
        
    except Exception as e:
        print(f"Error in finish_interview: {e}")
        return jsonify({"error": "Failed to finish interview"}), 500

def score_complete_interview(interview_data):
    """Score the complete interview with detailed analysis"""
    responses = interview_data.get('responses', {})
    scores = {}
    total_score = 0
    scored_questions = 0
    
    for key, response_data in responses.items():
        try:
            question = response_data.get('question', '')
            answer = response_data.get('answer', '')
            category = response_data.get('question_type', 'technical')
            
            # Determine difficulty from interview settings
            difficulty = interview_data['candidate_info'].get('difficulty', 'medium')
            if difficulty == 'mixed':
                difficulty = 'medium'  # Default for scoring
            
            # Score the answer
            evaluation = interview_system.score_answer_with_ai(question, answer, category, difficulty)
            
            scores[key] = {
                'question': question,
                'answer': answer[:200] + "..." if len(answer) > 200 else answer,
                'score': evaluation['score'],
                'feedback': evaluation['feedback'],
                'strengths': evaluation['strengths'],
                'improvements': evaluation['improvements'],
                'category': category
            }
            
            total_score += evaluation['score']
            scored_questions += 1
            
        except Exception as e:
            print(f"Error scoring {key}: {e}")
            scores[key] = {
                'score': 0,
                'feedback': 'Error during evaluation',
                'strengths': [],
                'improvements': ['Technical error occurred during scoring']
            }
    
    # Calculate summary
    if scored_questions > 0:
        final_score = round(total_score / scored_questions, 1)
        percentage = round((final_score / 10) * 100, 1)
    else:
        final_score = 0
        percentage = 0
    
    scores['summary'] = {
        'final_score': final_score,
        'percentage': percentage,
        'total_questions': scored_questions,
        'total_possible': scored_questions * 10 if scored_questions > 0 else 50
    }
    
    return scores

def generate_detailed_report(interview_data, scores, filename):
    """Generate a comprehensive PDF report"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(51, 51, 153)
        pdf.cell(0, 15, 'AI Interview Assessment Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Candidate Information
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, 'Candidate Information', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        candidate_info = interview_data.get('candidate_info', {})
        info_lines = [
            f"Name: {candidate_info.get('name', 'N/A')}",
            f"Position: {candidate_info.get('position', 'N/A')}",
            f"Experience: {candidate_info.get('experience', 'N/A')}",
            f"Interview Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Difficulty Level: {candidate_info.get('difficulty', 'N/A').title()}"
        ]
        
        for line in info_lines:
            pdf.cell(0, 8, line, 0, 1)
        
        pdf.ln(10)
        
        # Overall Score
        if 'summary' in scores:
            summary = scores['summary']
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Overall Performance', 0, 1)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f"Final Score: {summary['final_score']}/10 ({summary['percentage']}%)", 0, 1)
            pdf.cell(0, 8, f"Questions Answered: {summary['total_questions']}", 0, 1)
            pdf.ln(5)
        
        # Detailed Question Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Detailed Question Analysis', 0, 1)
        pdf.ln(5)
        
        for key, details in scores.items():
            if key != 'summary' and isinstance(details, dict):
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, f"Question {key.replace('q_', '')}: {details.get('category', 'N/A').title()}", 0, 1)
                
                pdf.set_font('Arial', '', 10)
                
                # Question
                question = details.get('question', 'N/A')
                pdf.multi_cell(0, 6, f"Q: {question[:150]}{'...' if len(question) > 150 else ''}")
                
                # Score
                pdf.cell(0, 6, f"Score: {details.get('score', 0)}/10", 0, 1)
                
                # Feedback
                feedback = details.get('feedback', 'No feedback available')
                pdf.multi_cell(0, 6, f"Feedback: {feedback[:200]}{'...' if len(feedback) > 200 else ''}")
                
                pdf.ln(5)
        
        pdf.output(filename)
        print(f"Report generated: {filename}")
        
    except Exception as e:
        print(f"Error generating report: {e}")

def send_interview_report_email(recipient_email, candidate_name, report_filename):
    """Send interview report via email"""
    try:
        sender_email = os.getenv('EMAIL_FROM')
        sender_password = os.getenv('EMAIL_PASS')
        
        if not sender_email or not sender_password:
            print("Email credentials not configured")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Your AI Interview Report - {candidate_name}"
        
        # Email body
        body = f"""
        Dear {candidate_name},
        
        Thank you for participating in our AI-powered interview assessment!
        
        Please find your detailed interview report attached. The report includes:
        - Overall performance score and analysis
        - Question-by-question feedback
        - Strengths and areas for improvement
        - Recommendations for skill development
        
        We appreciate your time and effort in completing this assessment.
        
        Best regards,
        AI Interview System Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF report
        if os.path.exists(report_filename):
            with open(report_filename, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(report_filename)}'
                )
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

@app.route('/download_report')
def download_report():
    """Download the latest interview report"""
    try:
        report_filename = session.get('last_report')
        if report_filename and os.path.exists(report_filename):
            return send_file(report_filename, as_attachment=True)
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        print(f"Error downloading report: {e}")
        return jsonify({"error": "Failed to download report"}), 500

if __name__ == '__main__':
    print("ðŸš€ Starting Advanced AI Interview System 2025...")
    print("ðŸ“‹ Features: Web Scraping, Voice Synthesis, Advanced Scoring, Email Reports")
    print("ðŸ¤– Using: Ollama + Hugging Face (Free Models)")
    print("ðŸŒ Server starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)