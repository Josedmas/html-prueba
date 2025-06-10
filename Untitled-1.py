#!/usr/bin/env python3
"""
NailedJob - Aplicación Web para Optimización de Currículums con IA
Implementación en Python usando FastAPI
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import json
import tempfile
import uuid
from pathlib import Path

# Imports para procesamiento de PDFs y IA
import PyPDF2
import google.generativeai as genai
from pymongo import MongoClient
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import requests
from bs4 import BeautifulSoup
import re

# Configuración de la aplicación
app = FastAPI(title="NailedJob", description="Aplicación web para optimización de currículums con IA")

# Configurar archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Variables de configuración (normalmente desde .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "tu_api_key_aqui")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "nailedjob")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "candidates")

# Configurar Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Configurar MongoDB (opcional)
try:
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client[MONGODB_DB_NAME]
    candidates_collection = db[MONGODB_COLLECTION_NAME]
    MONGODB_ENABLED = True
except Exception as e:
    print(f"MongoDB no disponible: {e}")
    MONGODB_ENABLED = False

# Modelos Pydantic
class JobOffer(BaseModel):
    title: str
    company: str
    description: str
    requirements: str
    url: Optional[str] = None

class Candidate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    experience: List[str] = []
    education: List[str] = []
    skills: List[str] = []
    languages: List[str] = []

class AnalysisRequest(BaseModel):
    job_offer: JobOffer
    resume_text: str
    language: str = "es"  # "es" o "en"

class AnalysisResult(BaseModel):
    compatibility_score: float
    strengths: List[str]
    weaknesses: List[str]
    extracted_candidate: Candidate
    analysis_text: str

class OptimizedResumeRequest(BaseModel):
    job_offer: JobOffer
    original_resume: str
    language: str = "es"

class OptimizedResumeResult(BaseModel):
    optimized_resume: str
    improvements: List[str]
    new_compatibility_score: float
    modifications_explanation: str

class JobSearchRequest(BaseModel):
    optimized_resume: str
    location: str = "España"
    language: str = "es"

class JobSearchResult(BaseModel):
    job_title: str
    company: str
    location: str
    url: str
    description: str

# Utilidades para procesamiento de texto
class TextProcessor:
    @staticmethod
    def extract_pdf_text(pdf_file) -> str:
        """Extrae texto de un archivo PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar PDF: {str(e)}")

    @staticmethod
    def extract_content_from_url(url: str) -> str:
        """Extrae contenido de una URL de oferta de trabajo"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            # Remover scripts y estilos
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text()
        except Exception as e:
            return f"Error al extraer contenido de la URL: {str(e)}"

# Servicios de IA
class AIService:
    def __init__(self):
        self.model = model

    def analyze_compatibility(self, job_offer: JobOffer, resume_text: str, language: str) -> AnalysisResult:
        """Analiza la compatibilidad entre una oferta de trabajo y un currículum"""
        
        prompt_templates = {
            "es": """
            Analiza la compatibilidad entre esta oferta de trabajo y el currículum proporcionado.
            
            OFERTA DE TRABAJO:
            Título: {job_title}
            Empresa: {company}
            Descripción: {description}
            Requisitos: {requirements}
            
            CURRÍCULUM:
            {resume_text}
            
            Por favor, proporciona:
            1. Una puntuación de compatibilidad del 0 al 100
            2. Lista de fortalezas del candidato para este puesto
            3. Lista de debilidades o áreas de mejora
            4. Extracción de datos del candidato (nombre, email, experiencia, educación, habilidades)
            
            Responde en formato JSON con esta estructura:
            {{
                "compatibility_score": número,
                "strengths": [lista de fortalezas],
                "weaknesses": [lista de debilidades],
                "candidate_data": {{
                    "name": "nombre",
                    "email": "email",
                    "phone": "teléfono",
                    "experience": [lista de experiencias],
                    "education": [lista de educación],
                    "skills": [lista de habilidades],
                    "languages": [lista de idiomas]
                }},
                "analysis_text": "análisis detallado"
            }}
            """,
            "en": """
            Analyze the compatibility between this job offer and the provided resume.
            
            JOB OFFER:
            Title: {job_title}
            Company: {company}
            Description: {description}
            Requirements: {requirements}
            
            RESUME:
            {resume_text}
            
            Please provide:
            1. A compatibility score from 0 to 100
            2. List of candidate strengths for this position
            3. List of weaknesses or areas for improvement
            4. Extraction of candidate data (name, email, experience, education, skills)
            
            Respond in JSON format with this structure:
            {{
                "compatibility_score": number,
                "strengths": [list of strengths],
                "weaknesses": [list of weaknesses],
                "candidate_data": {{
                    "name": "name",
                    "email": "email",
                    "phone": "phone",
                    "experience": [list of experiences],
                    "education": [list of education],
                    "skills": [list of skills],
                    "languages": [list of languages]
                }},
                "analysis_text": "detailed analysis"
            }}
            """
        }
        
        prompt = prompt_templates[language].format(
            job_title=job_offer.title,
            company=job_offer.company,
            description=job_offer.description,
            requirements=job_offer.requirements,
            resume_text=resume_text
        )
        
        try:
            response = self.model.generate_content(prompt)
            # Limpiar la respuesta para obtener solo el JSON
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result_data = json.loads(response_text)
            
            # Crear el objeto Candidate
            candidate_data = result_data.get("candidate_data", {})
            candidate = Candidate(
                name=candidate_data.get("name", ""),
                email=candidate_data.get("email", "usuario@email.com"),
                phone=candidate_data.get("phone"),
                experience=candidate_data.get("experience", []),
                education=candidate_data.get("education", []),
                skills=candidate_data.get("skills", []),
                languages=candidate_data.get("languages", [])
            )
            
            return AnalysisResult(
                compatibility_score=result_data["compatibility_score"],
                strengths=result_data["strengths"],
                weaknesses=result_data["weaknesses"],
                extracted_candidate=candidate,
                analysis_text=result_data["analysis_text"]
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en análisis de IA: {str(e)}")

    def optimize_resume(self, job_offer: JobOffer, original_resume: str, language: str) -> OptimizedResumeResult:
        """Optimiza un currículum para una oferta de trabajo específica"""
        
        prompt_templates = {
            "es": """
            Optimiza este currículum para que sea más compatible con la siguiente oferta de trabajo:
            
            OFERTA DE TRABAJO:
            Título: {job_title}
            Empresa: {company}
            Descripción: {description}
            Requisitos: {requirements}
            
            CURRÍCULUM ORIGINAL:
            {original_resume}
            
            Por favor, genera:
            1. Un currículum optimizado que mantenga la veracidad pero resalte las competencias relevantes
            2. Lista de mejoras realizadas
            3. Nueva puntuación de compatibilidad estimada
            4. Explicación de las modificaciones
            
            Responde en formato JSON:
            {{
                "optimized_resume": "currículum optimizado completo",
                "improvements": [lista de mejoras],
                "new_compatibility_score": número,
                "modifications_explanation": "explicación de cambios"
            }}
            """,
            "en": """
            Optimize this resume to be more compatible with the following job offer:
            
            JOB OFFER:
            Title: {job_title}
            Company: {company}
            Description: {description}
            Requirements: {requirements}
            
            ORIGINAL RESUME:
            {original_resume}
            
            Please generate:
            1. An optimized resume that maintains truthfulness but highlights relevant competencies
            2. List of improvements made
            3. New estimated compatibility score
            4. Explanation of modifications
            
            Respond in JSON format:
            {{
                "optimized_resume": "complete optimized resume",
                "improvements": [list of improvements],
                "new_compatibility_score": number,
                "modifications_explanation": "explanation of changes"
            }}
            """
        }
        
        prompt = prompt_templates[language].format(
            job_title=job_offer.title,
            company=job_offer.company,
            description=job_offer.description,
            requirements=job_offer.requirements,
            original_resume=original_resume
        )
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result_data = json.loads(response_text)
            
            return OptimizedResumeResult(
                optimized_resume=result_data["optimized_resume"],
                improvements=result_data["improvements"],
                new_compatibility_score=result_data["new_compatibility_score"],
                modifications_explanation=result_data["modifications_explanation"]
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en optimización de IA: {str(e)}")

    def search_jobs(self, optimized_resume: str, location: str, language: str) -> List[JobSearchResult]:
        """Busca ofertas de trabajo relevantes basadas en el currículum optimizado"""
        
        prompt_templates = {
            "es": """
            Basándote en este currículum optimizado, genera 10 ofertas de trabajo relevantes en {location}.
            
            CURRÍCULUM:
            {resume}
            
            Genera ofertas realistas que podrían encontrarse en InfoJobs, LinkedIn o Indeed.
            Incluye variedad de empresas y posiciones.
            
            Responde en formato JSON:
            {{
                "jobs": [
                    {{
                        "job_title": "título del puesto",
                        "company": "nombre de la empresa",
                        "location": "ubicación",
                        "url": "https://ejemplo.com/oferta",
                        "description": "descripción breve"
                    }}
                ]
            }}
            """,
            "en": """
            Based on this optimized resume, generate 10 relevant job offers in {location}.
            
            RESUME:
            {resume}
            
            Generate realistic offers that could be found on InfoJobs, LinkedIn or Indeed.
            Include variety of companies and positions.
            
            Respond in JSON format:
            {{
                "jobs": [
                    {{
                        "job_title": "job title",
                        "company": "company name",
                        "location": "location",
                        "url": "https://example.com/job",
                        "description": "brief description"
                    }}
                ]
            }}
            """
        }
        
        prompt = prompt_templates[language].format(
            resume=optimized_resume,
            location=location
        )
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            result_data = json.loads(response_text)
            
            jobs = []
            for job_data in result_data.get("jobs", []):
                jobs.append(JobSearchResult(
                    job_title=job_data["job_title"],
                    company=job_data["company"],
                    location=job_data["location"],
                    url=job_data["url"],
                    description=job_data["description"]
                ))
            
            return jobs
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en búsqueda de empleos: {str(e)}")

# Servicio de generación de PDFs
class PDFGenerator:
    @staticmethod
    def generate_resume_pdf(resume_text: str, candidate_name: str, output_path: str) -> str:
        """Genera un PDF profesional del currículum"""
        try:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Título
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            story.append(Paragraph(candidate_name, title_style))
            story.append(Spacer(1, 12))
            
            # Contenido del currículum
            content_style = ParagraphStyle(
                'CustomContent',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=6
            )
            
            # Dividir el texto en párrafos
            paragraphs = resume_text.split('\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para, content_style))
                    story.append(Spacer(1, 6))
            
            doc.build(story)
            return output_path
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generando PDF: {str(e)}")

# Inicializar servicios
text_processor = TextProcessor()
ai_service = AIService()
pdf_generator = PDFGenerator()

# Endpoints de la API

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze-compatibility")
async def analyze_compatibility(request: AnalysisRequest):
    """Analiza la compatibilidad entre una oferta de trabajo y un currículum"""
    try:
        result = ai_service.analyze_compatibility(
            request.job_offer,
            request.resume_text,
            request.language
        )
        
        # Guardar candidato en MongoDB si está habilitado
        if MONGODB_ENABLED:
            try:
                candidate_data = result.extracted_candidate.dict()
                candidate_data["created_at"] = datetime.now()
                candidate_data["job_offer"] = request.job_offer.dict()
                candidate_data["compatibility_score"] = result.compatibility_score
                candidates_collection.insert_one(candidate_data)
            except Exception as e:
                print(f"Error guardando en MongoDB: {e}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize-resume")
async def optimize_resume(request: OptimizedResumeRequest):
    """Optimiza un currículum para una oferta específica"""
    try:
        result = ai_service.optimize_resume(
            request.job_offer,
            request.original_resume,
            request.language
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search-jobs")
async def search_jobs(request: JobSearchRequest):
    """Busca ofertas de trabajo relevantes"""
    try:
        jobs = ai_service.search_jobs(
            request.optimized_resume,
            request.location,
            request.language
        )
        return {"jobs": jobs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Subir y procesar un archivo PDF de currículum"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    try:
        # Leer el archivo PDF
        content = await file.read()
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extraer texto
        with open(tmp_file_path, 'rb') as pdf_file:
            text = text_processor.extract_pdf_text(pdf_file)
        
        # Limpiar archivo temporal
        os.unlink(tmp_file_path)
        
        return {"text": text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {str(e)}")

@app.post("/api/extract-url-content")
async def extract_url_content(url: str = Form(...)):
    """Extrae contenido de una URL de oferta de trabajo"""
    try:
        content = text_processor.extract_content_from_url(url)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-resume-pdf")
async def generate_resume_pdf(
    resume_text: str = Form(...),
    candidate_name: str = Form(...)
):
    """Genera un PDF del currículum optimizado"""
    try:
        # Crear directorio temporal si no existe
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Generar nombre único para el archivo
        filename = f"resume_{uuid.uuid4().hex}.pdf"
        output_path = temp_dir / filename
        
        # Generar PDF
        pdf_generator.generate_resume_pdf(resume_text, candidate_name, str(output_path))
        
        return FileResponse(
            str(output_path),
            media_type="application/pdf",
            filename=f"{candidate_name}_resume.pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Endpoint de salud"""
    return {
        "status": "healthy",
        "mongodb_enabled": MONGODB_ENABLED,
        "timestamp": datetime.now().isoformat()
    }

# Configuración adicional
if __name__ == "__main__":
    import uvicorn
    
    # Crear directorios necesarios
    Path("temp").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    print("🚀 Iniciando NailedJob...")
    print("📝 Aplicación de optimización de currículums con IA")
    print(f"🔧 MongoDB: {'✅ Habilitado' if MONGODB_ENABLED else '❌ Deshabilitado'}")
    print("🌐 Servidor disponible en: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)