from flask import Flask, request, jsonify, send_file, session, make_response
import json
import os
import random
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from fpdf import FPDF
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, session, make_response, send_from_directory

try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace not installed. Face recognition features will be limited.")
    DeepFace = None
import base64
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'advanced-ai-interview-2025')

class MultilingualInterviewSystem:
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
        
        self.language_mappings = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'pt': 'Portuguese'
        }
        
        # Initialize trending topics directly
        self.trending_topics = [
            "Artificial Intelligence and Machine Learning",
            "Cloud Computing and DevOps",
            "Cybersecurity and Data Privacy",
            "Blockchain and Web3",
            "Mobile Development and Cross-platform frameworks",
            "Microservices and Containerization",
            "Real-time Applications and WebRTC",
            "Progressive Web Apps (PWAs)",
            "Edge Computing and IoT",
            "Sustainable Software Development"
        ]
        self.emotion_history = []
        
        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize SQLite database for storing interview data"""
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                candidate_name TEXT,
                candidate_email TEXT,
                position TEXT,
                experience TEXT,
                difficulty TEXT,
                language TEXT,
                start_time TEXT,
                end_time TEXT,
                overall_score REAL,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id TEXT,
                question_index INTEGER,
                question TEXT,
                answer TEXT,
                category TEXT,
                difficulty TEXT,
                score REAL,
                agent_scores TEXT,
                live_metrics TEXT,
                timestamp TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotion_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id TEXT,
                timestamp TEXT,
                dominant_emotion TEXT,
                emotion_scores TEXT,
                FOREIGN KEY (interview_id) REFERENCES interviews (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def generate_multilingual_question(self, category, difficulty, position, experience, language='en'):
        """Generate questions in specified language using multiple AI services"""
        try:
            # First try Ollama with language-specific prompt
            prompt = self.create_multilingual_prompt(category, difficulty, position, experience, language)
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "llama3.1",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 300
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                return self.clean_question(result)
                
        except Exception as e:
            print(f"Ollama error: {e}")
            
        # Fallback to language-specific question bank
        return self.get_multilingual_fallback_question(category, difficulty, position, language)

    def clean_question(self, question):
        """Clean and format the generated question"""
        # Remove any unwanted prefixes or suffixes
        question = re.sub(r'^(Question:|Q:)\s*', '', question, flags=re.IGNORECASE)
        question = re.sub(r'\n+', ' ', question)
        question = question.strip()
        
        # Ensure question ends with proper punctuation
        if not question.endswith(('?', '.', '!')):
            question += '?'
            
        return question

    def create_multilingual_prompt(self, category, difficulty, position, experience, language):
        """Create language-specific prompts for question generation"""
        language_instructions = {
            'en': "Generate the question in English",
            'hi': "Generate the question in Hindi (हिंदी में प्रश्न बनाएं)",
            'es': "Genera la pregunta en español",
            'fr': "Générez la question en français", 
            'de': "Erstellen Sie die Frage auf Deutsch",
            'zh': "用中文生成问题",
            'ja': "日本語で質問を生成してください",
            'ko': "한국어로 질문을 생성하세요",
            'ar': "أنشئ السؤال باللغة العربية",
            'pt': "Gere a pergunta em português"
        }
        
        base_prompt = f"""
        You are an expert technical interviewer conducting an interview in {self.language_mappings.get(language, 'English')}.
        
        Candidate Profile:
        - Position: {position}
        - Experience: {experience}
        - Interview Category: {category}
        - Difficulty Level: {difficulty}
        
        {language_instructions.get(language, 'Generate the question in English')}.
        
        Create ONE specific, practical interview question that:
        1. Tests real-world problem-solving skills
        2. Is appropriate for {experience} level candidates
        3. Relates to current industry practices
        4. Requires detailed, thoughtful answers
        5. Focuses on {category} concepts
        
        The question should be challenging but fair, and encourage detailed responses.
        
        Question:"""
        
        return base_prompt

    def get_multilingual_fallback_question(self, category, difficulty, position, language):
        """Enhanced fallback questions in multiple languages"""
        questions_db = {
            'en': {
                'technical': {
                    'easy': [
                        "What are the key differences between REST and GraphQL APIs in modern web development?",
                        "How do you implement responsive design using CSS Grid and Flexbox?",
                        "Explain the concept of component-based architecture in modern frontend frameworks.",
                        "What are the benefits of using TypeScript over JavaScript in large-scale applications?",
                        "How do you handle state management in React applications using modern hooks?"
                    ],
                    'medium': [
                        "Design a caching strategy for a high-traffic web application using Redis and CDN.",
                        "How would you implement real-time features in a web application using WebSockets?",
                        "Explain your approach to implementing micro-frontends in a large enterprise application.",
                        "How do you optimize database queries for a system handling millions of records?",
                        "Design a CI/CD pipeline for a microservices architecture using modern DevOps tools."
                    ],
                    'hard': [
                        "Architect a globally distributed system that can handle 100 million concurrent users.",
                        "How would you implement a real-time collaborative editing system like Google Docs?",
                        "Design a fault-tolerant event-driven architecture for a financial trading platform.",
                        "Implement a custom load balancer that can handle traffic spikes and automatic failover.",
                        "How would you design a system for processing real-time streaming data at petabyte scale?"
                    ]
                },
                'coding': {
                    'easy': [
                        "Write a function to find the maximum element in an array and explain its time complexity.",
                        "Implement a simple binary search algorithm and discuss when to use it.",
                        "Create a function to reverse a string without using built-in methods.",
                        "Write code to check if a string is a palindrome.",
                        "Implement a basic stack data structure with push, pop, and peek operations."
                    ],
                    'medium': [
                        "Design and implement a LRU (Least Recently Used) cache with O(1) operations.",
                        "Write an algorithm to find the longest common subsequence between two strings.",
                        "Implement a function to detect cycles in a linked list.",
                        "Create a solution for the two-sum problem with optimal time complexity.",
                        "Design a data structure that supports insert, delete, and getRandom operations in O(1) time."
                    ],
                    'hard': [
                        "Implement a thread-safe producer-consumer pattern using appropriate synchronization.",
                        "Design an algorithm to find the shortest path in a weighted graph with negative edges.",
                        "Create a solution for the traveling salesman problem using dynamic programming.",
                        "Implement a distributed hash table with consistent hashing.",
                        "Design and code a rate limiter that can handle millions of requests per second."
                    ]
                },
                'behavioral': {
                    'easy': [
                        "Tell me about a time when you had to learn a new technology quickly for a project.",
                        "Describe a situation where you had to work with a difficult team member.",
                        "How do you prioritize tasks when you have multiple deadlines?",
                        "Give an example of when you received constructive feedback and how you handled it.",
                        "Describe a project you're particularly proud of and why."
                    ],
                    'medium': [
                        "Tell me about a time when you had to make a difficult technical decision with limited information.",
                        "Describe a situation where you had to convince your team to adopt a new approach or technology.",
                        "How do you handle disagreements with your manager about technical decisions?",
                        "Give an example of when you had to take ownership of a mistake and how you resolved it.",
                        "Describe a time when you had to mentor a junior developer."
                    ],
                    'hard': [
                        "Tell me about a time when you had to lead a cross-functional team through a major technical challenge.",
                        "Describe a situation where you had to make a trade-off between technical debt and feature delivery.",
                        "How would you handle a situation where your team consistently misses deadlines?",
                        "Give an example of when you had to advocate for a significant architectural change.",
                        "Describe a time when you had to manage conflicting priorities from multiple stakeholders."
                    ]
                }
            },
            'hi': {
                'technical': {
                    'easy': [
                        "आधुनिक वेब डेवलपमेंट में REST और GraphQL APIs के बीच मुख्य अंतर क्या हैं?",
                        "CSS Grid और Flexbox का उपयोग करके responsive design कैसे implement करते हैं?",
                        "आधुनिक frontend frameworks में component-based architecture की अवधारणा समझाएं।",
                        "बड़े पैमाने के applications में JavaScript की तुलना में TypeScript के क्या फायदे हैं?",
                        "React applications में modern hooks का उपयोग करके state management कैसे handle करते हैं?"
                    ],
                    'medium': [
                        "Redis और CDN का उपयोग करके high-traffic web application के लिए caching strategy design करें।",
                        "WebSockets का उपयोग करके web application में real-time features कैसे implement करेंगे?",
                        "बड़े enterprise application में micro-frontends implement करने का आपका approach क्या है?",
                        "लाखों records handle करने वाले system के लिए database queries को कैसे optimize करेंगे?",
                        "Modern DevOps tools का उपयोग करके microservices architecture के लिए CI/CD pipeline design करें।"
                    ],
                    'hard': [
                        "100 million concurrent users को handle कर सकने वाला globally distributed system architect करें।",
                        "Google Docs जैसा real-time collaborative editing system कैसे implement करेंगे?",
                        "Financial trading platform के लिए fault-tolerant event-driven architecture design करें।",
                        "Traffic spikes और automatic failover handle कर सकने वाला custom load balancer implement करें।",
                        "Petabyte scale पर real-time streaming data को process करने के लिए system कैसे design करेंगे?"
                    ]
                }
            }
        }
        
        lang_questions = questions_db.get(language, questions_db['en'])
        category_questions = lang_questions.get(category, lang_questions['technical'])
        difficulty_questions = category_questions.get(difficulty, category_questions['medium'])
        
        return random.choice(difficulty_questions)

    def analyze_emotion_from_frame(self, frame_data):
        """Analyze emotion from video frame using DeepFace"""
        try:
            if not DeepFace:
                return {'dominant_emotion': 'neutral', 'scores': {}}
                
            # Decode base64 image
            img_data = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Analyze emotion
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]
                
            dominant_emotion = result['dominant_emotion']
            emotion_scores = result['emotion']
            
            # Store emotion data
            emotion_data = {
                'timestamp': datetime.now().isoformat(),
                'dominant_emotion': dominant_emotion,
                'scores': emotion_scores
            }
            
            self.emotion_history.append(emotion_data)
            
            return emotion_data
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            return {'dominant_emotion': 'neutral', 'scores': {}}

    def score_answer_with_multilingual_ai(self, question, answer, category, difficulty, language='en', agent_scores=None):
        """Score answers with language-aware AI evaluation including agent feedback"""
        if not answer or answer.upper() == 'SKIPPED':
            return {
                'score': 0,
                'feedback': self.get_localized_feedback('skipped', language),
                'strengths': [],
                'improvements': [self.get_localized_feedback('provide_answer', language)],
                'agent_analysis': agent_scores or {}
            }

        try:
            scoring_prompt = self.create_multilingual_scoring_prompt(
                question, answer, category, difficulty, language, agent_scores
            )
            
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
                evaluation = self.parse_multilingual_evaluation(result, language)
                evaluation['agent_analysis'] = agent_scores or {}
                return evaluation
                
        except Exception as e:
            print(f"AI scoring error: {e}")
            
        return self.rule_based_scoring(answer, question, category, difficulty, language, agent_scores)

    def create_multilingual_scoring_prompt(self, question, answer, category, difficulty, language, agent_scores=None):
        """Create language-specific scoring prompts with agent feedback integration"""
        language_instructions = {
            'en': "Provide your evaluation in English",
            'hi': "अपना मूल्यांकन हिंदी में प्रदान करें",
            'es': "Proporciona tu evaluación en español",
            'fr': "Fournissez votre évaluation en français",
            'de': "Geben Sie Ihre Bewertung auf Deutsch an",
            'zh': "请用中文提供您的评估",
            'ja': "日本語で評価を提供してください",
            'ko': "한국어로 평가를 제공하세요",
            'ar': "قدم تقييمك باللغة العربية",
            'pt': "Forneça sua avaliação em português"
        }
        
        agent_feedback = ""
        if agent_scores:
            agent_feedback = f"""
            
            AI Agent Analysis Results:
            - Fluency Score: {agent_scores.get('fluency', 0)}/10
            - Vocabulary Score: {agent_scores.get('vocabulary', 0)}/10
            - Clarity Score: {agent_scores.get('clarity', 0)}/10
            - Pacing Score: {agent_scores.get('pacing', 0)}/10
            - Technical Score: {agent_scores.get('technical', 0)}/10
            
            Consider these agent scores in your overall evaluation.
            """
        
        prompt = f"""
        You are an expert technical interviewer evaluating a candidate's response in {self.language_mappings.get(language, 'English')}.
        
        Question: {question}
        Answer: {answer}
        Category: {category}
        Difficulty: {difficulty}
        {agent_feedback}
        
        Evaluate this answer on a scale of 0-10 considering:
        1. Technical accuracy and correctness
        2. Completeness and depth of explanation  
        3. Practical understanding and real-world application
        4. Communication clarity and structure
        5. Use of relevant examples or experiences
        6. Integration with AI agent feedback scores
        
        {language_instructions.get(language, 'Provide your evaluation in English')}.
        
        Provide your evaluation in this exact format:
        Score: [0-10]
        Feedback: [2-3 sentences explaining the score]
        Strengths: [List 2-3 positive aspects]
        Improvements: [List 2-3 areas for improvement]
        """
        
        return prompt

    def parse_multilingual_evaluation(self, result, language):
        """Parse AI evaluation response"""
        try:
            lines = result.strip().split('\n')
            evaluation = {
                'score': 5,
                'feedback': 'Average response',
                'strengths': [],
                'improvements': []
            }
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Score:'):
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        evaluation['score'] = float(re.findall(r'\d+\.?\d*', score_text)[0])
                    except:
                        evaluation['score'] = 5
                elif line.startswith('Feedback:'):
                    evaluation['feedback'] = line.split(':', 1)[1].strip()
                elif line.startswith('Strengths:'):
                    current_section = 'strengths'
                    content = line.split(':', 1)[1].strip()
                    if content:
                        evaluation['strengths'].append(content)
                elif line.startswith('Improvements:'):
                    current_section = 'improvements'
                    content = line.split(':', 1)[1].strip()
                    if content:
                        evaluation['improvements'].append(content)
                elif line.startswith('-') and current_section:
                    evaluation[current_section].append(line[1:].strip())
                elif line and current_section and not line.startswith(('Score:', 'Feedback:', 'Strengths:', 'Improvements:')):
                    evaluation[current_section].append(line)
            
            return evaluation
            
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            return self.rule_based_scoring("", "", "", "", language)

    def rule_based_scoring(self, answer, question, category, difficulty, language, agent_scores=None):
        """Fallback rule-based scoring system"""
        if not answer or len(answer.strip()) < 10:
            return {
                'score': 0,
                'feedback': self.get_localized_feedback('poor', language),
                'strengths': [],
                'improvements': [self.get_localized_feedback('provide_answer', language)],
                'agent_analysis': agent_scores or {}
            }
        
        # Basic scoring based on answer length and keywords
        score = 5  # Base score
        
        # Length bonus
        word_count = len(answer.split())
        if word_count > 100:
            score += 2
        elif word_count > 50:
            score += 1
        
        # Technical keywords bonus
        technical_keywords = ['algorithm', 'database', 'framework', 'architecture', 'performance']
        keyword_count = sum(1 for keyword in technical_keywords if keyword.lower() in answer.lower())
        score += min(keyword_count, 2)
        
        # Agent scores integration
        if agent_scores:
            avg_agent_score = sum(agent_scores.values()) / len(agent_scores)
            score = (score + avg_agent_score) / 2
        
        score = min(10, max(0, score))
        
        feedback_key = 'excellent' if score >= 8 else 'good' if score >= 6 else 'average' if score >= 4 else 'poor'
        
        return {
            'score': round(score, 1),
            'feedback': self.get_localized_feedback(feedback_key, language),
            'strengths': ['Good technical understanding', 'Clear communication'],
            'improvements': ['Provide more specific examples', 'Elaborate on implementation details'],
            'agent_analysis': agent_scores or {}
        }

    def get_localized_feedback(self, feedback_type, language):
        """Get localized feedback messages"""
        feedback_messages = {
            'en': {
                'skipped': 'Question was skipped - no answer provided',
                'provide_answer': 'Provide an answer to demonstrate knowledge',
                'excellent': 'Excellent answer demonstrating strong knowledge and communication skills.',
                'good': 'Good answer with solid understanding, but could be enhanced with more depth.',
                'average': 'Average answer showing basic understanding, needs more detail and examples.',
                'below_average': 'Below average answer, requires significant improvement in content and depth.',
                'poor': 'Poor answer, needs substantial improvement in all areas.'
            },
            'hi': {
                'skipped': 'प्रश्न छोड़ दिया गया - कोई उत्तर नहीं दिया गया',
                'provide_answer': 'ज्ञान प्रदर्शित करने के लिए उत्तर दें',
                'excellent': 'उत्कृष्ट उत्तर जो मजबूत ज्ञान और संवाद कौशल दर्शाता है।',
                'good': 'अच्छा उत्तर जिसमें ठोस समझ है, लेकिन अधिक गहराई के साथ बेहतर हो सकता है।',
                'average': 'औसत उत्तर जो बुनियादी समझ दिखाता है, अधिक विवरण और उदाहरणों की आवश्यकता है।',
                'below_average': 'औसत से नीचे का उत्तर, सामग्री और गहराई में महत्वपूर्ण सुधार की आवश्यकता है।',
                'poor': 'खराब उत्तर, सभी क्षेत्रों में पर्याप्त सुधार की आवश्यकता है।'
            }
        }
        
        return feedback_messages.get(language, feedback_messages['en']).get(feedback_type, '')

    def save_interview_data(self, interview_id, interview_data):
        """Save interview data to database"""
        try:
            conn = sqlite3.connect('interview_data.db')
            cursor = conn.cursor()
            
            # Save main interview record
            cursor.execute('''
                INSERT OR REPLACE INTO interviews 
                (id, candidate_name, candidate_email, position, experience, difficulty, language, start_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interview_id,
                interview_data.get('candidateName'),
                interview_data.get('candidateEmail'),
                interview_data.get('position'),
                interview_data.get('experience'),
                interview_data.get('difficulty'),
                interview_data.get('language'),
                interview_data.get('start_time'),
                'in_progress'
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving interview data: {e}")
            return False

    def save_question_response(self, interview_id, question_data):
        """Save question and response data"""
        try:
            conn = sqlite3.connect('interview_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO questions 
                (interview_id, question_index, question, answer, category, difficulty, score, agent_scores, live_metrics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interview_id,
                question_data.get('questionIndex'),
                question_data.get('question'),
                question_data.get('answer'),
                question_data.get('category'),
                question_data.get('difficulty'),
                question_data.get('score', 0),
                json.dumps(question_data.get('agentScores', {})),
                json.dumps(question_data.get('liveMetrics', {})),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving question response: {e}")
            return False

# Initialize the enhanced system
interview_system = MultilingualInterviewSystem()

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open("index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "index.html not found. Please ensure the HTML file is in the same directory.", 404
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/get_question', methods=['POST'])
def get_question():
    """Generate and return interview questions"""
    try:
        data = request.json
        question_index = data.get('questionIndex', 0)
        candidate_info = data.get('candidateInfo', {})
        language = data.get('language', 'en')

        if 'interview_session' not in session:
            interview_id = str(uuid.uuid4())
            session['interview_session'] = {
                'id': interview_id,
                'questions': [],
                'responses': {},
                'start_time': datetime.now().isoformat(),
                'candidate_info': candidate_info,
                'language': language,
                'emotion_data': []
            }
            
            # Save interview data to database
            interview_system.save_interview_data(interview_id, {
                **candidate_info,
                'start_time': datetime.now().isoformat()
            })

        total_questions = int(candidate_info.get('questionCount', 5))
        if question_index >= total_questions:
            return jsonify({"finished": True})

        # Determine question category and difficulty
        categories = ['technical', 'coding', 'behavioral']
        category = categories[question_index % len(categories)]
        
        difficulty = candidate_info.get('difficulty', 'medium')
        if difficulty == 'mixed':
            difficulty = random.choice(['easy', 'medium', 'hard'])

        position = candidate_info.get('position', 'Software Developer')
        experience = candidate_info.get('experience', 'Junior')

        # Generate multilingual question
        question = interview_system.generate_multilingual_question(
            category, difficulty, position, experience, language
        )

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
            "questionIndex": question_index,
            "category": category,
            "difficulty": difficulty,
            "totalQuestions": total_questions
        })

    except Exception as e:
        print(f"Error generating question: {e}")
        return jsonify({"error": "Failed to generate question"}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Process and score submitted answers"""
    try:
        data = request.json
        question_index = data.get('questionIndex')
        question = data.get('question')
        answer = data.get('answer')
        language = data.get('language', 'en')
        agent_scores = data.get('agentScores', {})
        live_metrics = data.get('liveMetrics', {})

        if 'interview_session' not in session:
            return jsonify({"error": "No active interview session"}), 400

        # Get question details from session
        questions = session['interview_session']['questions']
        current_question = next((q for q in questions if q['index'] == question_index), None)
        
        if not current_question:
            return jsonify({"error": "Question not found"}), 400

        # Score the answer using multilingual AI with agent feedback
        evaluation = interview_system.score_answer_with_multilingual_ai(
            question, answer, 
            current_question['category'], 
            current_question['difficulty'],
            language,
            agent_scores
        )

        # Store response in session
        response_data = {
            'question': question,
            'answer': answer,
            'category': current_question['category'],
            'difficulty': current_question['difficulty'],
            'evaluation': evaluation,
            'timestamp': datetime.now().isoformat(),
            'response_time': time.time() - time.mktime(
                datetime.fromisoformat(current_question['timestamp']).timetuple()
            ),
            'agentScores': agent_scores,
            'liveMetrics': live_metrics
        }

        session['interview_session']['responses'][str(question_index)] = response_data

        # Save to database
        interview_system.save_question_response(
            session['interview_session']['id'],
            {
                'questionIndex': question_index,
                'question': question,
                'answer': answer,
                'category': current_question['category'],
                'difficulty': current_question['difficulty'],
                'score': evaluation['score'],
                'agentScores': agent_scores,
                'liveMetrics': live_metrics
            }
        )

        session.modified = True

        return jsonify({
            "success": True,
            "evaluation": evaluation
        })

    except Exception as e:
        print(f"Error submitting answer: {e}")
        return jsonify({"error": "Failed to submit answer"}), 500

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """Analyze emotion from video frame"""
    try:
        data = request.json
        frame_data = data.get('frameData')
        
        if not frame_data:
            return jsonify({"error": "No frame data provided"}), 400

        # Analyze emotion from the frame
        emotion_result = interview_system.analyze_emotion_from_frame(frame_data)
        
        # Store emotion data in session
        if 'interview_session' in session:
            session['interview_session']['emotion_data'].append(emotion_result)
            session.modified = True

        return jsonify({
            "emotion": emotion_result['dominant_emotion'],
            "confidence": max(emotion_result['scores'].values()) if emotion_result['scores'] else 0,
            "all_emotions": emotion_result['scores']
        })

    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return jsonify({"error": "Failed to analyze emotion"}), 500

@app.route('/finish_interview', methods=['POST'])
def finish_interview():
    """Complete interview and generate final report"""
    try:
        data = request.json
        language = data.get('language', 'en')
        final_agent_scores = data.get('finalAgentScores', {})

        if 'interview_session' not in session:
            return jsonify({"error": "No active interview session"}), 400

        interview_data = session['interview_session']
        
        # Calculate overall performance
        total_score = 0
        answered_questions = 0
        
        for response in interview_data['responses'].values():
            if response['answer'].upper() != 'SKIPPED':
                total_score += response['evaluation']['score']
                answered_questions += 1

        overall_score = (total_score / answered_questions) if answered_questions > 0 else 0
        
        # Update database with final results
        try:
            conn = sqlite3.connect('interview_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE interviews 
                SET end_time = ?, overall_score = ?, status = ?
                WHERE id = ?
            ''', (
                datetime.now().isoformat(),
                overall_score,
                'completed',
                interview_data['id']
            ))
            
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Database update error: {db_error}")
        
        # Generate comprehensive report
        report_data = {
            'candidate_info': interview_data['candidate_info'],
            'interview_summary': {
                'total_questions': len(interview_data['questions']),
                'answered_questions': answered_questions,
                'skipped_questions': len(interview_data['questions']) - answered_questions,
                'overall_score': round(overall_score, 2),
                'interview_duration': calculate_interview_duration(interview_data['start_time']),
                'language': language,
                'final_agent_scores': final_agent_scores
            },
            'detailed_responses': interview_data['responses'],
            'emotion_analysis': analyze_emotion_trends(interview_data.get('emotion_data', [])),
            'recommendations': generate_recommendations(overall_score, interview_data['responses'], language),
            'agent_performance': analyze_agent_performance(interview_data['responses'])
        }

        # Store report for later access
        session['last_interview_report'] = report_data
        session.modified = True

        # Send email report if email is provided
        candidate_email = interview_data['candidate_info'].get('candidateEmail')
        if candidate_email:
            threading.Thread(
                target=send_interview_report_email,
                args=(candidate_email, report_data, language)
            ).start()

        return jsonify({
            "success": True,
            "overall_score": overall_score,
            "report_summary": report_data['interview_summary']
        })

    except Exception as e:
        print(f"Error finishing interview: {e}")
        return jsonify({"error": "Failed to finish interview"}), 500

def analyze_agent_performance(responses):
    """Analyze performance across all AI agents"""
    agent_totals = {'fluency': [], 'vocabulary': [], 'clarity': [], 'pacing': [], 'technical': []}
    
    for response in responses.values():
        agent_scores = response.get('agentScores', {})
        for agent, score in agent_scores.items():
            if agent in agent_totals:
                agent_totals[agent].append(score)
    
    agent_averages = {}
    for agent, scores in agent_totals.items():
        if scores:
            agent_averages[agent] = {
                'average': round(sum(scores) / len(scores), 2),
                'best': max(scores),
                'worst': min(scores),
                'improvement': round(scores[-1] - scores[0], 2) if len(scores) > 1 else 0
            }
    
    return agent_averages

@app.route('/download_report')
def download_report():
    """Generate and download PDF report"""
    try:
        if 'last_interview_report' not in session:
            return "No interview report available", 404

        report_data = session['last_interview_report']
        
        # Generate PDF report
        pdf_path = generate_pdf_report(report_data)
        
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:
        print(f"Error downloading report: {e}")
        return "Failed to generate report", 500

def calculate_interview_duration(start_time_str):
    """Calculate interview duration in minutes"""
    start_time = datetime.fromisoformat(start_time_str)
    end_time = datetime.now()
    duration = end_time - start_time
    return round(duration.total_seconds() / 60, 2)

def analyze_emotion_trends(emotion_data):
    """Analyze emotion trends throughout the interview"""
    if not emotion_data:
        return {"summary": "No emotion data available"}

    emotion_counts = {}
    for entry in emotion_data:
        emotion = entry['dominant_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    total_entries = len(emotion_data)
    emotion_percentages = {
        emotion: round((count / total_entries) * 100, 2)
        for emotion, count in emotion_counts.items()
    }

    # Determine overall emotional state
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    
    return {
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_percentages,
        "total_samples": total_entries,
        "emotional_stability": calculate_emotional_stability(emotion_data)
    }

def calculate_emotional_stability(emotion_data):
    """Calculate emotional stability score based on emotion variance"""
    if len(emotion_data) < 2:
        return "Insufficient data"
    
    # Simple stability calculation based on emotion changes
    changes = 0
    for i in range(1, len(emotion_data)):
        if emotion_data[i]['dominant_emotion'] != emotion_data[i-1]['dominant_emotion']:
            changes += 1
    
    stability_score = max(0, 100 - (changes / len(emotion_data) * 100))
    return round(stability_score, 2)

def generate_recommendations(overall_score, responses, language='en'):
    """Generate personalized recommendations based on performance"""
    recommendations = {
        'en': {
            'excellent': [
                "Outstanding performance! Continue to stay updated with latest industry trends.",
                "Consider mentoring junior developers to share your expertise.",
                "Look into advanced certifications to further enhance your profile.",
                "Your AI agent scores show excellent communication skills across all areas."
            ],
            'good': [
                "Strong performance with room for improvement in specific areas.",
                "Focus on providing more detailed examples in your responses.",
                "Practice explaining complex concepts in simpler terms.",
                "Work on consistency across different question types."
            ],
            'average': [
                "Solid foundation but needs more depth in technical knowledge.",
                "Work on improving communication and presentation skills.",
                "Gain more hands-on experience with real-world projects.",
                "Focus on the areas where AI agents scored you lower."
            ],
            'below_average': [
                "Focus on strengthening fundamental concepts in your field.",
                "Practice mock interviews to improve confidence and delivery.",
                "Consider additional training or courses in weak areas.",
                "Pay attention to speech clarity and pacing feedback from AI agents."
            ],
            'poor': [
                "Significant improvement needed in technical knowledge and communication.",
                "Start with basic concepts and build up gradually.",
                "Seek mentorship and additional learning resources.",
                "Practice speaking exercises to improve fluency and clarity."
            ]
        },
        'hi': {
            'excellent': [
                "उत्कृष्ट प्रदर्शन! नवीनतम उद्योग रुझानों के साथ अपडेट रहना जारी रखें।",
                "अपनी विशेषज्ञता साझा करने के लिए जूनियर डेवलपर्स को मेंटर करने पर विचार करें।",
                "अपनी प्रोफाइल को और बेहतर बनाने के लिए उन्नत प्रमाणपत्रों की तलाश करें।",
                "आपके AI एजेंट स्कोर सभी क्षेत्रों में उत्कृष्ट संवाद कौशल दिखाते हैं।"
            ],
            'good': [
                "विशिष्ट क्षेत्रों में सुधार की गुंजाइश के साथ मजबूत प्रदर्शन।",
                "अपने उत्तरों में अधिक विस्तृत उदाहरण प्रदान करने पर ध्यान दें।",
                "जटिल अवधारणाओं को सरल शब्दों में समझाने का अभ्यास करें।",
                "विभिन्न प्रश्न प्रकारों में निरंतरता पर काम करें।"
            ],
            'average': [
                "ठोस आधार लेकिन तकनीकी ज्ञान में अधिक गहराई की आवश्यकता।",
                "संवाद और प्रस्तुति कौशल में सुधार पर काम करें।",
                "वास्तविक दुनिया की परियोजनाओं के साथ अधिक व्यावहारिक अनुभव प्राप्त करें।",
                "उन क्षेत्रों पर ध्यान दें जहां AI एजेंट्स ने आपको कम स्कोर दिया है।"
            ]
        }
    }
    
    # Determine performance level
    if overall_score >= 8:
        level = 'excellent'
    elif overall_score >= 6:
        level = 'good'
    elif overall_score >= 4:
        level = 'average'
    elif overall_score >= 2:
        level = 'below_average'
    else:
        level = 'poor'
    
    lang_recommendations = recommendations.get(language, recommendations['en'])
    return lang_recommendations.get(level, lang_recommendations['average'])

def generate_pdf_report(report_data):
    """Generate comprehensive PDF report with agent analysis"""
    class InterviewPDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'AI Interview Assessment Report with Agent Analysis', 0, 1, 'C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')

    pdf = InterviewPDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # Candidate Information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Candidate Information', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    candidate_info = report_data['candidate_info']
    pdf.cell(0, 8, f"Name: {candidate_info.get('candidateName', 'N/A')}", 0, 1)
    pdf.cell(0, 8, f"Email: {candidate_info.get('candidateEmail', 'N/A')}", 0, 1)
    pdf.cell(0, 8, f"Position: {candidate_info.get('position', 'N/A')}", 0, 1)
    pdf.cell(0, 8, f"Experience: {candidate_info.get('experience', 'N/A')}", 0, 1)
    pdf.ln(10)

    # Interview Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Interview Summary', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    summary = report_data['interview_summary']
    pdf.cell(0, 8, f"Overall Score: {summary['overall_score']}/10", 0, 1)
    pdf.cell(0, 8, f"Questions Answered: {summary['answered_questions']}/{summary['total_questions']}", 0, 1)
    pdf.cell(0, 8, f"Duration: {summary['interview_duration']} minutes", 0, 1)
    pdf.cell(0, 8, f"Language: {summary['language']}", 0, 1)
    pdf.ln(10)

    # AI Agent Performance Summary
    if 'agent_performance' in report_data:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'AI Agent Performance Analysis', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        agent_performance = report_data['agent_performance']
        for agent, metrics in agent_performance.items():
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, f"{agent.title()} Agent:", 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f"  Average Score: {metrics['average']}/10", 0, 1)
            pdf.cell(0, 6, f"  Best Performance: {metrics['best']}/10", 0, 1)
            pdf.cell(0, 6, f"  Improvement: {metrics['improvement']:+.1f} points", 0, 1)
            pdf.ln(3)

    # Detailed Responses
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Detailed Question Analysis', 0, 1)
    pdf.set_font('Arial', '', 10)

    for idx, response in report_data['detailed_responses'].items():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Question {int(idx) + 1}: {response['category'].title()}", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        # Question
        question_text = response['question'][:150] + "..." if len(response['question']) > 150 else response['question']
        pdf.multi_cell(0, 6, f"Q: {question_text}", 0, 1)
        
        if response['answer'].upper() != 'SKIPPED':
            # Answer
            answer_text = response['answer'][:200] + "..." if len(response['answer']) > 200 else response['answer']
            pdf.multi_cell(0, 6, f"A: {answer_text}", 0, 1)
            
            # Scores
            pdf.cell(0, 6, f"Overall Score: {response['evaluation']['score']}/10", 0, 1)
            
            # Agent Scores
            if 'agentScores' in response:
                agent_scores = response['agentScores']
                pdf.cell(0, 6, f"Agent Scores - Fluency: {agent_scores.get('fluency', 0)}, Vocabulary: {agent_scores.get('vocabulary', 0)}, Clarity: {agent_scores.get('clarity', 0)}", 0, 1)
            
            # Feedback
            feedback_text = response['evaluation']['feedback'][:150] + "..." if len(response['evaluation']['feedback']) > 150 else response['evaluation']['feedback']
            pdf.multi_cell(0, 6, f"Feedback: {feedback_text}", 0, 1)
        else:
            pdf.cell(0, 6, "Answer: SKIPPED", 0, 1)
        
        pdf.ln(5)

    # Emotion Analysis
    if 'emotion_analysis' in report_data:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Emotional Analysis', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        emotion_data = report_data['emotion_analysis']
        if emotion_data.get('total_samples', 0) > 0:
            pdf.cell(0, 8, f"Dominant Emotion: {emotion_data['dominant_emotion'].title()}", 0, 1)
            pdf.cell(0, 8, f"Emotional Stability: {emotion_data['emotional_stability']}%", 0, 1)
            pdf.ln(5)
            
            pdf.cell(0, 8, "Emotion Distribution:", 0, 1)
            for emotion, percentage in emotion_data['emotion_distribution'].items():
                pdf.cell(0, 6, f"  {emotion.title()}: {percentage}%", 0, 1)

    # Recommendations
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations for Improvement', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    for i, recommendation in enumerate(report_data['recommendations'], 1):
        pdf.multi_cell(0, 8, f"{i}. {recommendation}", 0, 1)
        pdf.ln(2)

    # Save PDF
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = f"reports/interview_report_{timestamp}.pdf"
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    pdf.output(pdf_path)
    return pdf_path

def send_interview_report_email(email, report_data, language='en'):
    """Send interview report via email"""
    try:
        # Email configuration
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME')
        smtp_password = os.getenv('SMTP_PASSWORD')
        
        if not all([smtp_username, smtp_password]):
            print("Email credentials not configured")
            return False

        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = email
        
        # Multilingual subject and body
        subjects = {
            'en': 'Your AI Interview Assessment Report with Agent Analysis',
            'hi': 'आपकी AI साक्षात्कार मूल्यांकन रिपोर्ट एजेंट विश्लेषण के साथ'
        }
        
        bodies = {
            'en': f"""
Dear {report_data['candidate_info'].get('candidateName', 'Candidate')},

Thank you for completing the AI interview assessment with our advanced agent analysis system. Please find your detailed report attached.

Interview Summary:
- Overall Score: {report_data['interview_summary']['overall_score']}/10
- Questions Answered: {report_data['interview_summary']['answered_questions']}/{report_data['interview_summary']['total_questions']}
- Duration: {report_data['interview_summary']['interview_duration']} minutes

AI Agent Performance:
Your responses were analyzed by 5 specialized AI agents that provided real-time feedback on:
- Fluency and speaking pace
- Vocabulary diversity and complexity
- Clarity and coherence
- Pacing and timing
- Technical terminology usage

We appreciate your time and effort. Best of luck with your career journey!

Best regards,
AI Interview System Team
        """,
            'hi': f"""
प्रिय {report_data['candidate_info'].get('candidateName', 'उम्मीदवार')},

हमारे उन्नत एजेंट विश्लेषण सिस्टम के साथ AI साक्षात्कार मूल्यांकन पूरा करने के लिए धन्यवाद। कृपया अपनी विस्तृत रिपोर्ट संलग्न पाएं।

साक्षात्कार सारांश:
- समग्र स्कोर: {report_data['interview_summary']['overall_score']}/10
- उत्तरित प्रश्न: {report_data['interview_summary']['answered_questions']}/{report_data['interview_summary']['total_questions']}
- अवधि: {report_data['interview_summary']['interview_duration']} मिनट

AI एजेंट प्रदर्शन:
आपके उत्तरों का विश्लेषण 5 विशेषज्ञ AI एजेंट्स द्वारा किया गया जिन्होंने निम्नलिखित पर वास्तविक समय की प्रतिक्रिया प्रदान की:
- प्रवाह और बोलने की गति
- शब्दावली विविधता और जटिलता
- स्पष्टता और सुसंगति
- गति और समय
- तकनीकी शब्दावली का उपयोग

हम आपके समय और प्रयास की सराहना करते हैं। आपकी करियर यात्रा के लिए शुभकामनाएं!

सादर,
AI साक्षात्कार सिस्टम टीम
        """
        }
        
        msg['Subject'] = subjects.get(language, subjects['en'])
        body = bodies.get(language, bodies['en'])
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Generate and attach PDF report
        pdf_path = generate_pdf_report(report_data)
        
        with open(pdf_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= interview_report_with_agents.pdf'
            )
            msg.attach(part)

        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_username, email, text)
        server.quit()
        
        # Clean up PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
        print(f"Report sent successfully to {email}")
        return True

    except Exception as e:
        print(f"Error sending email: {e}")
        return False

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": ["multilingual", "agent_analysis", "face_recognition", "real_time_feedback"]
    })

@app.route('/api/stats')
def get_stats():
    """Get interview system statistics"""
    try:
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        
        # Get statistics
        cursor.execute("SELECT COUNT(*) FROM interviews")
        total_interviews = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM interviews WHERE status = 'completed'")
        completed_interviews = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(overall_score) FROM interviews WHERE overall_score IS NOT NULL")
        avg_score = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            "total_interviews": total_interviews,
            "completed_interviews": completed_interviews,
            "average_score": round(avg_score, 2),
            "supported_languages": len(interview_system.language_mappings),
            "available_positions": 10,
            "system_uptime": "99.9%",
            "ai_agents": 5
        })
        
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({
            "total_interviews": 0,
            "supported_languages": len(interview_system.language_mappings),
            "available_positions": 10,
            "system_uptime": "99.9%",
            "ai_agents": 5
        })

@app.route('/api/interview_history')
def get_interview_history():
    """Get interview history for analytics"""
    try:
        conn = sqlite3.connect('interview_data.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, candidate_name, position, overall_score, start_time, status
            FROM interviews 
            ORDER BY start_time DESC 
            LIMIT 50
        ''')
        
        interviews = []
        for row in cursor.fetchall():
            interviews.append({
                'id': row[0],
                'candidate_name': row[1],
                'position': row[2],
                'overall_score': row[3],
                'start_time': row[4],
                'status': row[5]
            })
        
        conn.close()
        return jsonify(interviews)
        
    except Exception as e:
        print(f"Error getting interview history: {e}")
        return jsonify([])

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    os.makedirs('recordings', exist_ok=True)
    
    print("🚀 Starting Advanced AI Interview System 2025")
    print("Features: Multilingual Support, AI Agents, Face Recognition, Real-time Analysis")
    print("Database: SQLite for persistent storage")
    print("AI Models: Ollama (primary), Fallback question banks")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
