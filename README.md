# Multilingual AI Interview Simulator 2025

## Project Description

**Multilingual AI Interview Simulator 2025** is a full-stack AI-powered interview system that replicates real-world technical interviews. It features automated question generation, real-time scoring, multilingual support, emotion detection, and detailed PDF reports.

This project is ideal for job seekers, hiring teams, and educational platforms aiming to conduct dynamic and insightful mock interviews.

---

## Features

- Custom interview setup with role, experience level, difficulty, and language
- Question generation via **Ollama** and **Hugging Face** (free models)
- Multilingual support (English, Hindi, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Portuguese)
- Real-time facial emotion detection using **DeepFace**
- AI agents for analyzing:
  - Fluency
  - Vocabulary
  - Clarity
  - Pacing
  - Technical depth
- Automated scoring and answer feedback
- PDF report generation and email delivery
- Animated, responsive HTML/CSS front-end

---

## Technologies Used

### Backend

- Python
- Flask
- SQLite3
- DeepFace
- OpenCV
- FPDF
- Requests
- SMTP (email delivery)
- dotenv

### Frontend

- HTML / CSS / JavaScript
- Responsive animated interface
- Language selector and modal UI

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Indrajeet-Badhel/MULTI_AGENTIC_INTERVIEW_COMPANION.git
   cd MULTI_AGENTIC_INTERVIEW_COMPANION

 2. **Install dependency**
     ```bash
     pip install -r requirements.txt
3.**Set up environment variables**
   Create a .env file:
```bash
 SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SECRET_KEY=your_secret_key
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
HUGGINGFACE_API_KEY=your_huggingface_key
OLLAMA_BASE_URL=http://localhost:11434
```
4. **Run Ollama (if installed)**
   ```bash
   ollama pull llama3.1
   ollama run llama3.1
5. **Start the app**
   python interview_app.py
   
6.**Access the platform**
Open your browser at: http://localhost:5000
   
## Project Structure
```bash
project-root/
├── interview_app.py         # Main backend logic
├── index.html               # Frontend UI
├── randggom.py              # Additional logic file
├── interview_data.db        # SQLite database
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
```

## how it works 
1.Candidate configures their interview preferences

2.Questions are generated via AI (Ollama / Hugging Face)

3.Real-time emotion tracking and scoring occurs during the interview

4.Each answer is scored using AI and agent-based metrics

5.Final performance is compiled into a downloadable PDF

6.Report is automatically emailed to the candidate

## License
This project is intended for educational and research purposes only. Usage of third-party APIs or models is subject to their individual licenses.

## Author
Developed by Indrajeet Badhel, 2025
``` bash 
---

Let me know if you want:

- Badges (Python version, license, build status)
- A live demo badge or section
- Deployment guide (e.g., Docker or Streamlit version)
```



   
   
