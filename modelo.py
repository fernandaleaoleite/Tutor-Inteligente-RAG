import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import logging

# Configuração inicial
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Configuração de logging
logging.basicConfig(
    filename='conversa.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S'
)

class SistemaChatEducativo:
    def __init__(self):
        # Verifica se a chave de API foi fornecida
        if not api_key:
            raise ValueError("GEMINI_API_KEY não está configurada!")

        # Modelo principal (tutor)
        self.llm_tutor = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            google_api_key=api_key
        )
        
        # Juiz (modelo mais conservador)
        self.llm_juiz = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        
        # Configuração do RAGc
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model="models/embedding-001"
        )
        
        # Carrega documentos para o RAG
        self.carregar_documentos("documentos_conhecimento")
        
        # Prompt do sistema para o tutor
        self.prompt_tutor = """
        Você é um tutor educacional especializado em tecnologia e ciência. 
        Suas respostas devem ser:
        - Baseadas nos documentos fornecidos (quando disponíveis)
        - Clara e didática
        - Com exemplos práticos quando possível
        - Adaptada ao nível do estudante
        - Incluir sugestões para aprofundamento
        
        Se a informação não estiver nos documentos, seja transparente sobre isso.
        """
        
        # Prompt do juiz
        self.prompt_juiz = """
        Você é um avaliador crítico de respostas educacionais. Avalie:
        1. Precisão técnica (0-10)
        2. Clareza (0-10)
        3. Adequação ao nível do aluno (0-10)
        4. Utilidade dos exemplos (0-10)
        5. Sugestões de aprofundamento (0-10)
        
        Formato da resposta:
        ✅/⚠️ [Aprovado/Reprovado]
        Pontuação: X/50
        Análise: [análise detalhada]
        Melhorias: [sugestões]
        """
    
    def carregar_documentos(self, pasta):
        """Carrega documentos para o sistema RAG"""
        try:
            # Carrega todos os arquivos .txt da pasta
            documentos = []
            for nome in os.listdir(pasta):
                if nome.endswith(".txt"):
                    caminho = os.path.join(pasta, nome)
                    loader = TextLoader(caminho, encoding="utf-8")
                    documentos.extend(loader.load())
            
            # Divide os documentos em chunks
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs_divididos = splitter.split_documents(documentos)
            
            # Cria o índice FAISS
            self.db = FAISS.from_documents(docs_divididos, self.embeddings)
            
            # Cria a cadeia RAG
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm_tutor,
                retriever=self.db.as_retriever(),
                return_source_documents=True
            )
            
            logging.info(f"Sistema RAG carregado com {len(documentos)} documentos")
            return True
        
        except Exception as e:
            logging.error(f"Erro ao carregar documentos: {str(e)}")
            return False
    
    def responder_pergunta(self, pergunta):
        """Processa uma pergunta usando RAG e valida com juiz"""
        try:
            # Passo 1: Busca com RAG
            resposta_rag = self.rag_chain.invoke({"query": pergunta})
            resposta_texto = resposta_rag['result']
            fontes = [doc.metadata['source'] for doc in resposta_rag['source_documents']]
            
            # Passo 2: Validação com juiz
            mensagens_juiz = [
                SystemMessage(content=self.prompt_juiz),
                HumanMessage(content=f"Pergunta: {pergunta}\nResposta: {resposta_texto}")
            ]
            avaliacao = self.llm_juiz.invoke(mensagens_juiz).content
            
            # Passo 3: Log da conversa
            registro = {
                'timestamp': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
                'pergunta': pergunta,
                'resposta': resposta_texto,
                'fontes': fontes,
                'avaliacao': avaliacao
            }
            
            logging.info(f"PERGUNTA: {pergunta}")
            logging.info(f"RESPOSTA: {resposta_texto}")
            logging.info(f"FONTES: {', '.join(fontes)}")
            logging.info(f"AVALIAÇÃO: {avaliacao}")
            
            # Formata a resposta final
            resposta_final = f"""
            🎓 Resposta Educacional:
            
            {resposta_texto}
            
            📚 Fontes consultadas:
            {', '.join(fontes) if fontes else 'Nenhuma fonte específica consultada'}
            
            🔍 Avaliação da Resposta:
            {avaliacao}
            """
            
            return resposta_final
        
        except Exception as e:
            logging.error(f"Erro ao processar pergunta: {str(e)}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."

# Instância global do sistema
sistema_chat = SistemaChatEducativo()