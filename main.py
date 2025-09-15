#!/usr/bin/env python3

import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Existing imports
from imap_tools import MailBox, AND
from notion_client import Client

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_ufrn_langchain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS - Estruturas de dados tipadas
# ============================================================================

class EmailData(BaseModel):
    """Modelo para dados do email"""
    assunto: str
    remetente: str
    data_recebimento: Optional[str] = None
    conteudo_preview: str
    tem_anexos: bool = False
    quantidade_anexos: int = 0


class EmailClassification(BaseModel):
    """Modelo para classificação do email"""
    categoria: str = Field(description="Categoria do email (REUNIAO, ACADEMICO, etc.)")
    urgencia: str = Field(description="Nível de urgência (ALTA, MEDIA, BAIXA)")
    necessita_resposta: str = Field(description="Se precisa resposta (SIM, NAO, TALVEZ)")
    confianca: float = Field(description="Confiança na classificação (0-1)")
    justificativa: str = Field(description="Justificativa da classificação")
    resumo_executivo: str = Field(description="Resumo executivo do email")
    acao_sugerida: str = Field(description="Ação sugerida")
    prazo_estimado: str = Field(description="Prazo estimado para resposta")


class MeetingAnalysis(BaseModel):
    """Modelo para análise de reunião"""
    is_meeting: bool = Field(description="Se é uma reunião")
    confidence: float = Field(description="Confiança na detecção (0-1)")
    extracted_time: Optional[str] = Field(description="Horário extraído")
    extracted_date: Optional[str] = Field(description="Data extraída")
    meeting_type: Optional[str] = Field(description="Tipo de reunião")
    justification: str = Field(description="Justificativa da análise")


# ============================================================================
# OUTPUT PARSERS - Parsers para estruturar saídas do LLM
# ============================================================================

class EmailClassificationParser(BaseOutputParser[EmailClassification]):
    """Parser para classificação de email"""

    def parse(self, text: str) -> EmailClassification:
        try:
            lines = text.strip().split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    data[key] = value
            return EmailClassification(
                categoria=data.get('categoria', 'INFORMATIVO'),
                urgencia=data.get('urgencia', 'BAIXA'),
                necessita_resposta=data.get('necessita_resposta', 'TALVEZ'),
                confianca=float(data.get('confianca', '0.8')),
                justificativa=data.get('justificativa', 'Análise automática'),
                resumo_executivo=data.get('resumo_executivo', 'Email processado'),
                acao_sugerida=data.get('acao_sugerida', 'Avaliar contexto'),
                prazo_estimado=data.get('prazo_estimado', 'N/A')
            )
        except Exception as e:
            logger.warning(f"Erro no parse da classificação: {e}")
            return EmailClassification(
                categoria='INFORMATIVO', urgencia='BAIXA', necessita_resposta='TALVEZ',
                confianca=0.5, justificativa='Erro no processamento',
                resumo_executivo='Email com erro de processamento',
                acao_sugerida='Revisar manualmente', prazo_estimado='N/A'
            )


class MeetingAnalysisParser(BaseOutputParser[MeetingAnalysis]):
    """Parser para análise de reunião"""

    def parse(self, text: str) -> MeetingAnalysis:
        try:
            lines = text.strip().split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    data[key] = value
            return MeetingAnalysis(
                is_meeting=data.get('is_meeting', 'false').lower() == 'true',
                confidence=float(data.get('confidence', '0.5')),
                extracted_time=data.get('extracted_time'),
                extracted_date=data.get('extracted_date'),
                meeting_type=data.get('meeting_type'),
                justification=data.get('justification', 'Análise automática')
            )
        except Exception as e:
            logger.warning(f"Erro no parse da reunião: {e}")
            return MeetingAnalysis(is_meeting=False, confidence=0.5, justification='Erro no processamento')


# ============================================================================
# LANGCHAIN TOOLS - Ferramentas que o agente pode usar
# ============================================================================

class EmailClassifierTool(BaseTool):
    """Ferramenta para classificar emails e gerar resumos dinâmicos."""
    name: str = "email_classifier"
    description: str = "Classifica emails por categoria, urgência e necessidade de resposta, e gera resumos dinâmicos."
    llm: Any
    chain: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.chain = self._create_classification_chain()

    def _create_classification_chain(self):
        prompt = PromptTemplate(
            input_variables=["remetente", "assunto", "conteudo"],
            template="""
Você é um especialista em análise e classificação de emails institucionais da UFRN. Sua tarefa é analisar o email, classificá-lo e, em seguida, criar um resumo executivo e uma ação sugerida adaptados ao contexto da categoria.

DADOS DO EMAIL:
Remetente: {remetente}
Assunto: {assunto}
Conteúdo: {conteudo}

PASSO 1: CLASSIFICAÇÃO
Primeiro, analise e classifique o email de acordo com as seguintes regras:

CATEGORIAS DISPONÍVEIS:
- DOUTORADO: Comunicações sobre sua pesquisa, orientador, artigos, disciplinas e prazos do doutorado.
- REUNIAO: Agendamentos, convites ou remarcações de reuniões.
- ACADEMICO: Questões acadêmicas gerais, que NÃO são do doutorado.
- ADMINISTRATIVO: Processos, documentos, solicitações de RH, etc.
- FINANCEIRO: Bolsas, pagamentos, auxílios.
- URGENTE: Demandas críticas com prazo que não se encaixam nas outras categorias.
- INFORMATIVO: Comunicados, newsletters, avisos.
- PESSOAL: Comunicações não relacionadas ao trabalho.

URGÊNCIA (analisar o conteúdo para decidir):
- ALTA: Prazos curtos, solicitações diretas de superiores ou do orientador, correções urgentes em artigos, problemas de acesso.
- MEDIA: Demandas administrativas com prazo razoável, comunicados importantes, sugestões de leitura do orientador.
- BAIXA: Informativos, newsletters, comunicados gerais sem ação direta.

RESPOSTA:
- SIM: Se o email faz uma pergunta direta, solicita uma ação ou é de uma pessoa importante (orientador).
- NAO: Informativos automáticos, newsletters.
- TALVEZ: Contexto ambíguo.

PASSO 2: CRIAÇÃO DE RESUMO E AÇÃO DINÂMICOS
Após decidir a Categoria, use as seguintes diretrizes para criar o Resumo_executivo e a Acao_sugerida:

- PARA A CATEGORIA DOUTORADO:
  - Foco do Resumo: Qual é a demanda principal (revisão, prazo, feedback, artigo)? Quem a solicitou (orientador, co-autor, revista)?
  - Estilo da Ação Sugerida: Usar verbos acionáveis e específicos. Ex: "Revisar o artigo enviado pelo orientador", "Responder ao co-autor sobre o prazo de submissão", "Verificar o novo cronograma da disciplina XYZ".

- PARA A CATEGORIA REUNIAO:
  - Foco do Resumo: Quem está convidando, qual o tópico, e qual a data/hora proposta.
  - Estilo da Ação Sugerida: Ex: "Confirmar presença na reunião com X", "Propor novo horário para a reunião sobre Y", "Adicionar evento ao calendário".

- PARA A CATEGORIA ADMINISTRATIVO / FINANCEIRO:
  - Foco do Resumo: Qual é o processo ou documento mencionado e qual a principal informação ou pendência.
  - Estilo da Ação Sugerida: Ex: "Verificar o status do processo SEI", "Enviar documento solicitado pelo RH", "Confirmar recebimento da bolsa".

- PARA AS DEMAIS CATEGORIAS:
  - Foco do Resumo: Uma frase concisa sobre o tópico principal do email.
  - Estilo da Ação Sugerida: "Arquivar para referência", "Ler quando houver tempo" ou "Nenhuma ação necessária".

FORMATO DE RESPOSTA (exatamente assim):
Categoria: [CATEGORIA]
Urgencia: [ALTA/MEDIA/BAIXA]
Necessita_resposta: [SIM/NAO/TALVEZ]
Confianca: [0.0-1.0]
Justificativa: [Explicação da decisão, mencionando o porquê da urgência]
Resumo_executivo: [Resumo gerado conforme as diretrizes do PASSO 2]
Acao_sugerida: [Ação gerada conforme as diretrizes do PASSO 2]
Prazo_estimado: [Prazo para ação, se mencionado no email]
"""
        )
        return prompt | self.llm | EmailClassificationParser()

    def _run(self, email_data: str) -> str:
        try:
            email = json.loads(email_data)
            result = self.chain.invoke({
                "remetente": email['remetente'],
                "assunto": email['assunto'],
                "conteudo": email['conteudo_preview']
            })
            return result.model_dump_json()
        except Exception as e:
            logger.error(f"Erro na classificação: {e}")
            return json.dumps({"erro": str(e)})


class MeetingDetectorTool(BaseTool):
    """Ferramenta especializada em detectar reuniões"""
    name: str = "meeting_detector"
    description: str = "Detecta se um email é realmente sobre marcar uma reunião com alta precisão"
    llm: Any
    chain: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.chain = self._create_meeting_chain()

    def _create_meeting_chain(self):
        prompt = PromptTemplate(
            input_variables=["remetente", "assunto", "conteudo"],
            template="""
Você é um especialista em detectar reuniões em emails. Seja MUITO CRITERIOSO.
EMAIL:
De: {remetente}
Assunto: {assunto}
Conteúdo: {conteudo}
ANÁLISE PASSO A PASSO:
1. INTENÇÃO DE ENCONTRO:
   - Há proposta explícita de reunião/encontro?
   - Palavras-chave: "reunião", "conversar", "encontrar", "meeting"
   - NÃO considere: menções casuais, relatórios, documentos
2. PROPOSTA DE HORÁRIO:
   - Há horário específico proposto?
   - Formato: "às 10:30", "10h30", "10 horas"
   - Data: "amanhã", "segunda", "dia X"
3. CONTEXTO DO REMETENTE:
   - É pessoa física (não sistema automático)?
   - Tem autoridade para marcar reuniões?
4. TIPO DE CONTEÚDO:
   - É SOLICITAÇÃO/CONVITE ou apenas INFORMAÇÃO?
   - Newsletters, comunicados = NÃO são reuniões
   - Documentos circulares = NÃO são reuniões
CRITÉRIOS RÍGIDOS:
✅ SIM = Intenção + Horário + Pessoa física + Solicitação
❌ NÃO = Falta qualquer critério acima
FORMATO DE RESPOSTA:
Is_meeting: [true/false]
Confidence: [0.0-1.0]
Extracted_time: [horário encontrado ou null]
Extracted_date: [data encontrada ou null]
Meeting_type: [presencial/virtual/indefinido ou null]
Justification: [Explicação detalhada da decisão]
"""
        )
        return prompt | self.llm | MeetingAnalysisParser()

    def _run(self, email_data: str) -> str:
        try:
            email = json.loads(email_data)
            result = self.chain.invoke({
                "remetente": email['remetente'],
                "assunto": email['assunto'],
                "conteudo": email['conteudo_preview']
            })
            return result.model_dump_json()
        except Exception as e:
            logger.error(f"Erro na detecção de reunião: {e}")
            return json.dumps({"erro": str(e)})


# SUBSTITUA ESTA CLASSE INTEIRA NO SEU main.py

class NotionDashboardTool(BaseTool):
    """Ferramenta para adicionar ao dashboard Notion"""
    name: str = "notion_dashboard"
    description: str = "Adiciona emails classificados ao dashboard executivo no Notion"
    notion_client: Any
    database_id: str

    def _run(self, data: str) -> str:
        try:
            parsed_data = json.loads(data)
            email_data = parsed_data['email']
            classification = parsed_data['classification']
            meeting_analysis = parsed_data.get('meeting_analysis')

            # Determinar prioridade e tipo
            if classification.get('categoria') == 'DOUTORADO':
                prioridade = classification['urgencia']  # <<< CORREÇÃO: Usa a urgência vinda do LLM
                tipo = "DOUTORADO"
            elif meeting_analysis and meeting_analysis.get('is_meeting'):
                prioridade = "CRÍTICA"
                tipo = "REUNIÃO"
            elif '@ufrn.br' in email_data['remetente'] and classification['urgencia'] == 'ALTA':
                prioridade = "CRÍTICA"
                tipo = "COLABORADOR_UFRN"
            elif classification['urgencia'] == 'ALTA':
                prioridade = "ALTA"
                tipo = "EMAIL_AÇÃO"
            else:
                prioridade = classification['urgencia']
                tipo = "INFORMATIVO"

            # Monta o título com prefixos úteis
            prefixo = ""
            if tipo == "REUNIÃO":
                prefixo = "[REUNIÃO] "
            elif tipo == "DOUTORADO":
                prefixo = "[DOUTORADO] "

            titulo_final = f"{prefixo}Email: {email_data['assunto'][:70]}"

            properties = {
                "TÍTULO": {"title": [{"text": {"content": titulo_final}}]},
                "TIPO": {"select": {"name": tipo}},
                "PRIORIDADE": {"select": {"name": prioridade}},
                "STATUS": {"select": {"name": "NOVO"}},
                "DESCRIÇÃO": {"rich_text": [{"text": {
                    "content": f"De: {email_data['remetente']}\n\nResumo: {classification['resumo_executivo']}\n\nAção: {classification['acao_sugerida']}\n\nJustificativa: {classification['justificativa']}"}}]},
                "RESUMO_EXECUTIVO": {"rich_text": [{"text": {"content": classification['resumo_executivo']}}]},
                "NECESSITA_RESPOSTA": {"select": {"name": classification['necessita_resposta']}},
                "AGENTE_ORIGEM": {"rich_text": [{"text": {"content": "UFRN_LangChain_Agent"}}]},
                "DATA_CRIAÇÃO": {"date": {"start": datetime.now().isoformat()}}
            }
            response = self.notion_client.pages.create(parent={"database_id": self.database_id}, properties=properties)
            return f"Sucesso: {response.get('id', 'N/A')}"
        except Exception as e:
            # Esta linha é crucial. O erro exato será gravado no log.
            logger.error(f"Erro no Notion: {e}", exc_info=True)
            return f"Erro: {str(e)}"


# ============================================================================
# AGENT UFRN - Classe principal com LangChain
# ============================================================================

class UFRNEmailAgent:
    """Agente inteligente para processamento de emails da UFRN usando LangChain"""

    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                temperature=0.1,
                model_name="gpt-3.5-turbo",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            self.gmail_user = os.getenv('GMAIL_USER')
            self.gmail_password = os.getenv('GMAIL_PASSWORD')
            self.notion_client = Client(auth=os.getenv('NOTION_TOKEN'))
            self.notion_database_id = os.getenv('NOTION_DATABASE_ID')

            self.tools = [
                EmailClassifierTool(llm=self.llm),
                MeetingDetectorTool(llm=self.llm),
                NotionDashboardTool(notion_client=self.notion_client, database_id=self.notion_database_id)
            ]

            self.summary_chain = self._create_summary_chain()
            self.briefing_chain = self._create_briefing_chain()
            self.session_emails = []
            logger.info("🤖 Agent UFRN LangChain inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro na inicialização do agente: {e}")
            raise

    def _create_summary_chain(self):
        prompt = PromptTemplate(
            input_variables=["emails_data", "total_emails", "reunioes", "colaboradores_ufrn"],
            template="""
Você é um assistente executivo especializado em criar briefings profissionais.
DADOS DA SESSÃO:
Total de emails: {total_emails}
Reuniões detectadas: {reunioes}
Colaboradores UFRN: {colaboradores_ufrn}
EMAILS PROCESSADOS:
{emails_data}
INSTRUÇÕES:
Crie um resumo executivo profissional seguindo esta estrutura:
1. SITUAÇÃO GERAL (1-2 frases sobre volume e contexto)
2. PRIORIDADES CRÍTICAS (reuniões e demandas urgentes)
3. COLABORADORES UFRN (comunicações internas importantes)
4. AÇÕES NECESSÁRIAS (lista priorizada)
5. RECOMENDAÇÕES (próximos passos estratégicos)
ESTILO:
- Tom executivo e acionável
- Foco em decisões e prioridades
- Máximo 300 palavras
- Sem bullets, texto corrido
- Destaque pessoas por nome quando relevante
RESUMO EXECUTIVO:
"""
        )
        return prompt | self.llm | StrOutputParser()

    def _create_briefing_chain(self):
        """Cria uma chain para o briefing por categoria."""
        prompt = PromptTemplate(
            input_variables=["emails_data"],
            template="""
Você é um assistente executivo mestre em criar briefings diários. Sua tarefa é analisar uma lista de emails processados e criar um "Briefing por Categoria" conciso.

LISTA DE EMAILS PROCESSADOS (com sua classificação e resumo individual):
{emails_data}

INSTRUÇÕES PARA O BRIEFING:
1.  Para CADA uma das categorias abaixo, liste o nome da categoria.
2.  Ao lado do nome, coloque entre parênteses o número total de emails recebidos para essa categoria.
3.  Em seguida, escreva um resumo de UMA ÚNICA FRASE sobre o que eram esses emails, capturando a essência das mensagens.
4.  Se uma categoria não tiver nenhum email na lista, escreva "(0 emails): Nenhum email nesta categoria."
5.  Seja extremamente conciso e direto ao ponto.

CATEGORIAS A SEREM OBRIGATORIAMENTE LISTADAS:
- DOUTORADO
- REUNIAO
- ACADEMICO
- ADMINISTRATIVO
- FINANCEIRO
- URGENTE
- INFORMATIVO
- PESSOAL

EXEMPLO DE SAÍDA:
DOUTORADO (2 emails): Houve uma solicitação de revisão de artigo pelo orientador e uma confirmação de submissão.
FINANCEIRO (1 email): Recebido um lembrete de vencimento de fatura.
ADMINISTRATIVO (0 emails): Nenhum email nesta categoria.

BRIEFING POR CATEGORIA:
"""
        )
        return prompt | self.llm | StrOutputParser()

    def fetch_emails(self, limit: int = 50) -> List[EmailData]:
        try:
            with MailBox('imap.gmail.com').login(self.gmail_user, self.gmail_password) as mailbox:
                unread_emails = list(mailbox.fetch(AND(seen=False), limit=limit))
                if not unread_emails:
                    logger.info("📭 Nenhum email não lido")
                    return []
                logger.info(f"📧 Encontrados {len(unread_emails)} emails não lidos")
                emails = []
                for msg in unread_emails:
                    try:
                        email_data = EmailData(
                            assunto=msg.subject, remetente=msg.from_,
                            data_recebimento=msg.date.isoformat() if msg.date else None,
                            conteudo_preview=msg.text[:1500] if msg.text else "Sem conteúdo",
                            tem_anexos=len(msg.attachments) > 0,
                            quantidade_anexos=len(msg.attachments)
                        )
                        emails.append(email_data)
                    except Exception as e:
                        logger.warning(f"Erro ao processar email: {e}")
                return emails
        except Exception as e:
            logger.error(f"Erro ao buscar emails: {e}")
            return []

    def process_single_email(self, email: EmailData) -> Dict[str, Any]:
        """Processar um único email usando as ferramentas de forma orquestrada."""
        try:
            email_json = email.model_dump_json()
            logger.info(f"Iniciando processamento orquestrado do email: {email.assunto[:50]}")

            classifier_tool = self.tools[0]
            classification_result_str = classifier_tool.run(email_json)
            classification_data = json.loads(classification_result_str)
            logger.info(f"Classificação obtida: {classification_data.get('categoria')}")

            meeting_tool = self.tools[1]
            meeting_result_str = meeting_tool.run(email_json)
            meeting_data = json.loads(meeting_result_str)
            logger.info(f"Análise de reunião: {meeting_data.get('is_meeting')}")

            notion_tool = self.tools[2]
            payload_for_notion = {
                "email": email.model_dump(),
                "classification": classification_data,
                "meeting_analysis": meeting_data
            }
            notion_result = notion_tool.run(json.dumps(payload_for_notion, ensure_ascii=False))
            logger.info(f"Resultado do Notion: {notion_result}")

            is_meeting = meeting_data.get('is_meeting', False)

            return {
                "email": email.model_dump(),
                "classification": classification_data,
                "meeting_analysis": meeting_data,
                "notion_result": notion_result,
                "is_meeting": is_meeting,
                "processed": "Sucesso" in notion_result
            }
        except Exception as e:
            logger.error(f"Erro no processamento orquestrado do email: {e}", exc_info=True)
            return {"email": email.model_dump(), "agent_result": f"Erro: {str(e)}", "processed": False,
                    "is_meeting": False}

    def process_emails(self, limit: int = 50, batch_size: int = 3) -> List[Dict[str, Any]]:
        try:
            emails = self.fetch_emails(limit)
            if not emails:
                return []
            all_processed = []
            stats = {"total": len(emails), "reunioes": 0, "colaboradores_ufrn": 0, "dashboard_success": 0}

            for i in range(0, len(emails), batch_size):
                batch = emails[i:i + batch_size]
                logger.info(f"🔄 Processando lote {i // batch_size + 1}: {len(batch)} emails")
                for j, email in enumerate(batch):
                    email_num = i + j + 1
                    logger.info(f"📧 Processando email {email_num}: {email.assunto[:50]}...")
                    result = self.process_single_email(email)
                    if '@ufrn.br' in email.remetente:
                        stats["colaboradores_ufrn"] += 1
                    if result.get("is_meeting"):
                        stats["reunioes"] += 1
                    if result["processed"]:
                        stats["dashboard_success"] += 1
                    all_processed.append(result)
                    self.session_emails.append(result)
                    logger.info(f"✅ Email {email_num} processado")

            logger.info("📊 Gerando relatórios da sessão com LangChain...")
            reports = self.generate_session_reports(all_processed, stats)
            self.create_executive_page(reports, stats)

            logger.info(f"📊 Processamento LangChain concluído:")
            logger.info(f"   📧 Total: {stats['total']} emails")
            logger.info(f"   🗓️ Reuniões: {stats['reunioes']}")
            logger.info(f"   👥 Colaboradores UFRN: {stats['colaboradores_ufrn']}")
            logger.info(f"   📋 Dashboard: {stats['dashboard_success']}/{stats['total']} sucessos")

            print("\n" + "=" * 120)
            print(f"📋 RESUMO EXECUTIVO LANGCHAIN - {datetime.now().strftime('%d/%m/%Y %H:%M')}")
            print("=" * 120)
            print(reports['summary'])
            print("\n" + "=" * 120)
            print(f"📋 BRIEFING POR CATEGORIA")
            print("=" * 120)
            print(reports['briefing'])
            print("=" * 120)

            return all_processed
        except Exception as e:
            logger.error(f"Erro no processamento: {e}")
            return []

    def generate_session_reports(self, processed_emails: List[Dict], stats: Dict) -> Dict[str, str]:
        """Gerar o resumo executivo e o briefing por categoria usando LangChain."""
        try:
            emails_summary_list = []
            for res in processed_emails:
                email = res['email']
                classification = res['classification']
                line = (
                    f"Categoria: {classification['categoria']} | "
                    f"De: {email['remetente']} | "
                    f"Assunto: {email['assunto']} | "
                    f"Resumo: {classification['resumo_executivo']}"
                )
                emails_summary_list.append(line)

            emails_text = "\n---\n".join(emails_summary_list)

            executive_summary = self.summary_chain.invoke({
                "emails_data": emails_text,
                "total_emails": stats["total"],
                "reunioes": stats["reunioes"],
                "colaboradores_ufrn": stats["colaboradores_ufrn"]
            })

            categorical_briefing = self.briefing_chain.invoke({
                "emails_data": emails_text
            })

            return {
                "summary": executive_summary,
                "briefing": categorical_briefing
            }
        except Exception as e:
            logger.error(f"Erro ao gerar relatórios da sessão: {e}")
            return {
                "summary": f"Erro ao gerar resumo executivo. {stats['total']} emails processados.",
                "briefing": "Erro ao gerar briefing por categoria."
            }

    def create_executive_page(self, reports: Dict[str, str], stats: Dict):
        """Criar página executiva no Notion com ambos os relatórios."""
        try:
            current_hour = datetime.now().hour
            if 6 <= current_hour < 12:
                turno = "Manhã"
            elif 12 <= current_hour < 18:
                turno = "Tarde"
            else:
                turno = "Noite"
            page_title = f"Resumo LangChain - {datetime.now().strftime('%d/%m/%Y')} - {turno}"

            page_content = (
                f"{reports['summary']}\n\n---\n\n"
                f"BRIEFING POR CATEGORIA:\n{reports['briefing']}"
            )

            stats_content = (
                f"Resumo executivo gerado por LangChain Agent.\n\nEstatísticas:\n"
                f"- Total: {stats['total']}\n"
                f"- Colaboradores UFRN: {stats['colaboradores_ufrn']}\n"
                f"- Reuniões: {stats['reunioes']}\n"
                f"- Dashboard: {stats['dashboard_success']}/{stats['total']}"
            )

            page_data = {
                "parent": {"database_id": self.notion_database_id},
                "properties": {
                    "TÍTULO": {"title": [{"text": {"content": page_title}}]},
                    "TIPO": {"select": {"name": "RESUMO_EXECUTIVO"}},
                    "PRIORIDADE": {"select": {"name": "ALTA"}},
                    "STATUS": {"select": {"name": "ATIVO"}},
                    "RESUMO_EXECUTIVO": {"rich_text": [{"text": {"content": page_content}}]},
                    "DESCRIÇÃO": {"rich_text": [{"text": {"content": stats_content}}]},
                    "AGENTE_ORIGEM": {"rich_text": [{"text": {"content": "UFRN_LangChain_Agent"}}]},
                    "DATA_CRIAÇÃO": {"date": {"start": datetime.now().isoformat()}}
                }
            }
            response = self.notion_client.pages.create(**page_data)
            if response and response.get('id'):
                logger.info(f"📋 Página executiva LangChain criada: {response['id']}")
                print(f"📄 Resumo executivo LangChain salvo no Notion: {response['id']}")
        except Exception as e:
            logger.error(f"Erro ao criar página executiva: {e}")

    def save_results(self, processed_emails: List[Dict]) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emails_langchain_processados_{timestamp}.json"
            results = {
                "timestamp": datetime.now().isoformat(),
                "total_emails": len(processed_emails),
                "framework": "LangChain",
                "agent_type": "Orchestrated Tools",
                "emails": processed_emails
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Resultados LangChain salvos: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Erro ao salvar: {e}")
            return None


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    try:
        print("🤖 AGENT UFRN - LangChain Edition")
        print("=" * 50)
        print("🧠 Framework: LangChain (Modernized)")
        print("🔧 Agent Type: Orchestrated Tools")
        print("🛠️ Tools: EmailClassifier, MeetingDetector, NotionDashboard")
        print("=" * 50)
        agent = UFRNEmailAgent()
        processed_emails = agent.process_emails(limit=50, batch_size=3)
        if processed_emails:
            filename = agent.save_results(processed_emails)
            print(f"\n📈 RESUMO FINAL:")
            print(f"   📧 Emails processados: {len(processed_emails)}")
            print(f"   🧠 Framework: LangChain")
            print(f"   🤖 Agent: Orchestrated Tools")
            print(f"   💾 Arquivo: {filename}")
        print(f"\n✅ Processamento LangChain concluído!")
    except Exception as e:
        logger.error(f"Erro na execução: {e}")
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()
