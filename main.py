import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from imap_tools import MailBox, AND
from openai import OpenAI
from dataclasses import dataclass
from notion_client import Client

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_ufrn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuração dos hyperparâmetros do modelo"""
    # Modelo a ser usado
    model: str = "gpt-4.1-mini"

    # Parâmetros de geração para processamento em lote
    temperature: float = 0.2  # Baixa para consistência na classificação
    max_tokens: int = 3000  # Otimizado para emails UFRN
    top_p: float = 0.9  # Nucleus sampling
    frequency_penalty: float = 0.0  # Sem penalidade de frequência
    presence_penalty: float = 0.0  # Sem penalidade de presença

    # Configurações específicas
    batch_temperature: float = 0.1  # Mais determinístico
    test_temperature: float = 0.0  # Zero para testes de conexão

    def get_batch_params(self):
        """Retorna parâmetros otimizados para processamento em lote"""
        return {
            "model": self.model,
            "temperature": self.batch_temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

    def get_test_params(self):
        """Retorna parâmetros para teste de conexão"""
        return {
            "model": self.model,
            "temperature": self.test_temperature,
            "max_tokens": 10,
            "top_p": 1.0
        }

    def print_config(self):
        """Imprime configuração atual"""
        print(f"\n🔧 CONFIGURAÇÃO DO MODELO (PROCESSAMENTO EM LOTE):")
        print(f"  Modelo: {self.model}")
        print(f"  Temperature (lote): {self.batch_temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Top-p: {self.top_p}")
        print(f"  Frequency penalty: {self.frequency_penalty}")
        print(f"  Presence penalty: {self.presence_penalty}")


class AgentUFRN:
    def __init__(self, model_config: ModelConfig = None):
        """Inicializar o agente com configuração de modelo"""
        try:
            # Configuração do modelo
            self.model_config = model_config or ModelConfig()

            # Credenciais OpenAI
            self.openai_client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )

            # Credenciais Gmail
            self.gmail_user = os.getenv('GMAIL_USER')
            self.gmail_password = os.getenv('GMAIL_PASSWORD')

            # Credenciais Notion
            self.notion_client = Client(auth=os.getenv('NOTION_TOKEN'))
            self.notion_database_id = os.getenv('NOTION_DATABASE_ID')

            logger.info("Agente UFRN inicializado com sucesso")
            self.model_config.print_config()

        except Exception as e:
            logger.error(f"Erro ao inicializar agente: {e}")
            raise

    def classify_emails_batch(self, emails_data):
        """Classificar múltiplos emails em uma única chamada ao LLM"""
        try:
            # Preparar dados dos emails para o prompt
            emails_for_prompt = []
            for i, email in enumerate(emails_data, 1):
                email_text = f"""
EMAIL {i}:
Assunto: {email['assunto']}
Remetente: {email['remetente']}
Conteúdo: {email['conteudo_preview'][:800]}
Tem anexos: {'Sim' if email['tem_anexos'] else 'Não'}
---"""
                emails_for_prompt.append(email_text)

            emails_text = "\n".join(emails_for_prompt)

            prompt = f"""
Analise os seguintes {len(emails_data)} emails e classifique cada um deles.

CATEGORIAS DISPONÍVEIS:
1. ADMINISTRATIVO - questões burocráticas, documentos, prazos administrativos
2. ACADEMICO - questões de pesquisa, orientação, eventos acadêmicos, defesas
3. FINANCEIRO - bolsas, pagamentos, questões financeiras, prestação de contas
4. URGENTE - requer ação imediata (independente da categoria)
5. INFORMATIVO - apenas para conhecimento, newsletters, comunicados gerais
6. PESSOAL - emails pessoais não relacionados ao trabalho

EMAILS PARA CLASSIFICAR:
{emails_text}

Responda APENAS em formato JSON com um array contendo a classificação de cada email na ordem apresentada:

{{
  "classificacoes": [
    {{
      "email_numero": 1,
      "categoria": "CATEGORIA_PRINCIPAL",
      "urgencia": "ALTA/MEDIA/BAIXA",
      "resumo": "Breve resumo do conteúdo",
      "acao_sugerida": "Ação específica recomendada",
      "prazo_estimado": "Prazo para resposta/ação",
      "palavras_chave": ["palavra1", "palavra2", "palavra3"],
      "confianca": 0.95
    }},
    {{
      "email_numero": 2,
      "categoria": "CATEGORIA_PRINCIPAL",
      "urgencia": "ALTA/MEDIA/BAIXA",
      "resumo": "Breve resumo do conteúdo",
      "acao_sugerida": "Ação específica recomendada",
      "prazo_estimado": "Prazo para resposta/ação",
      "palavras_chave": ["palavra1", "palavra2", "palavra3"],
      "confianca": 0.90
    }}
  ],
  "processamento": {{
    "total_emails": {len(emails_data)},
    "timestamp": "{datetime.now().isoformat()}"
  }}
}}

IMPORTANTE: Mantenha a ordem dos emails e numere corretamente cada classificação.
"""

            # Usar parâmetros configurados para processamento em lote
            params = self.model_config.get_batch_params()

            logger.info(f"Enviando {len(emails_data)} emails para classificação em lote...")

            response = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                **params
            )

            classification_text = response.choices[0].message.content.strip()

            # Log dos parâmetros usados (apenas em debug)
            logger.debug(f"Parâmetros usados: {params}")

            # Tentar parsear JSON
            try:
                batch_result = json.loads(classification_text)

                if "classificacoes" not in batch_result:
                    raise ValueError("Resposta não contém campo 'classificacoes'")

                classificacoes = batch_result["classificacoes"]

                if len(classificacoes) != len(emails_data):
                    logger.warning(
                        f"Número de classificações ({len(classificacoes)}) diferente do número de emails ({len(emails_data)})")

                # Adicionar informações sobre o modelo usado
                for classificacao in classificacoes:
                    classificacao['modelo_usado'] = params['model']
                    classificacao['temperatura_usada'] = params['temperature']
                    classificacao['processamento_em_lote'] = True

                logger.info(f"✅ {len(classificacoes)} emails classificados em lote com sucesso")
                return classificacoes

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Erro ao parsear resposta JSON: {e}")
                logger.error(f"Resposta recebida: {classification_text[:500]}...")

                # Retornar classificações de erro para todos os emails
                error_classifications = []
                for i in range(len(emails_data)):
                    error_classifications.append({
                        "email_numero": i + 1,
                        "categoria": "ERRO_CLASSIFICACAO",
                        "urgencia": "MEDIA",
                        "resumo": "Erro ao processar classificação em lote",
                        "acao_sugerida": "Revisar manualmente",
                        "prazo_estimado": "N/A",
                        "palavras_chave": ["erro", "lote"],
                        "confianca": 0.0,
                        "modelo_usado": params['model'],
                        "temperatura_usada": params['temperature'],
                        "processamento_em_lote": True
                    })
                return error_classifications

        except Exception as e:
            logger.error(f"Erro ao classificar emails em lote: {e}")

            # Retornar classificações de erro para todos os emails
            error_classifications = []
            for i in range(len(emails_data)):
                error_classifications.append({
                    "email_numero": i + 1,
                    "categoria": "ERRO_SISTEMA",
                    "urgencia": "MEDIA",
                    "resumo": f"Erro do sistema: {str(e)}",
                    "acao_sugerida": "Verificar configurações",
                    "prazo_estimado": "N/A",
                    "palavras_chave": ["erro", "sistema"],
                    "confianca": 0.0,
                    "modelo_usado": self.model_config.model,
                    "temperatura_usada": self.model_config.batch_temperature,
                    "processamento_em_lote": True
                })
            return error_classifications

    def add_to_executive_dashboard(self, email_data, classification):
        """Adicionar item ao Dashboard Executivo no Notion"""
        try:
            # Só adiciona se for uma ação que precisa de atenção
            if classification.get('categoria') in ['ADMINISTRATIVO', 'ACADEMICO', 'FINANCEIRO', 'URGENTE']:

                # Determinar prioridade baseada na categoria e urgência
                prioridade = "CRÍTICA" if classification.get('categoria') == 'URGENTE' else "ALTA"

                # Determinar tipo de entrada
                tipo = "EMAIL_AÇÃO"

                # Criar entrada no Notion
                properties = {
                    "Título": {
                        "title": [
                            {
                                "text": {
                                    "content": f"Email: {email_data['assunto'][:100]}"
                                }
                            }
                        ]
                    },
                    "Tipo": {
                        "select": {
                            "name": tipo
                        }
                    },
                    "Prioridade": {
                        "select": {
                            "name": prioridade
                        }
                    },
                    "Status": {
                        "select": {
                            "name": "NOVO"
                        }
                    },
                    "Descrição": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": f"De: {email_data['remetente']}\n\nResumo: {classification.get('resumo', 'N/A')}\n\nAção: {classification.get('acao_sugerida', 'Revisar email')}"
                                }
                            }
                        ]
                    },
                    "Agente_Origem": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": "UFRN_v2"
                                }
                            }
                        ]
                    },
                    "Data_Criação": {
                        "date": {
                            "start": datetime.now().isoformat()
                        }
                    }
                }

                # Adicionar prazo se disponível
                if classification.get('prazo_estimado') and classification.get('prazo_estimado') != 'N/A':
                    # Calcular data baseada no prazo (simplificado)
                    prazo_dias = 7  # Default
                    if 'urgente' in classification.get('prazo_estimado', '').lower():
                        prazo_dias = 1
                    elif 'semana' in classification.get('prazo_estimado', '').lower():
                        prazo_dias = 7

                    prazo_date = datetime.now().replace(hour=23, minute=59, second=59)
                    from datetime import timedelta
                    prazo_date += timedelta(days=prazo_dias)

                    properties["Prazo"] = {
                        "date": {
                            "start": prazo_date.isoformat()
                        }
                    }

                # Criar página no Notion
                response = self.notion_client.pages.create(
                    parent={"database_id": self.notion_database_id},
                    properties=properties
                )

                logger.info(f"✅ Item adicionado ao Dashboard Executivo: {email_data['assunto'][:50]}...")
                return response

        except Exception as e:
            logger.error(f"❌ Erro ao adicionar ao Dashboard Executivo: {e}")
            return None

    def process_emails(self, limit=10, batch_size=2):
        """Processar emails não lidos em lotes otimizados"""
        try:
            with MailBox('imap.gmail.com').login(self.gmail_user, self.gmail_password) as mailbox:
                # Buscar emails não lidos (limitado)
                unread_emails = list(mailbox.fetch(AND(seen=False), limit=limit))

                if not unread_emails:
                    logger.info("Nenhum email não lido encontrado")
                    return []

                logger.info(f"Encontrados {len(unread_emails)} emails não lidos")

                all_processed_emails = []

                # Processar emails em lotes
                for batch_start in range(0, len(unread_emails), batch_size):
                    batch_end = min(batch_start + batch_size, len(unread_emails))
                    batch_emails = unread_emails[batch_start:batch_end]

                    logger.info(
                        f"Processando lote {batch_start // batch_size + 1}: emails {batch_start + 1}-{batch_end}")

                    # Extrair dados dos emails do lote
                    batch_data = []
                    for msg in batch_emails:
                        try:
                            email_data = {
                                "timestamp": datetime.now().isoformat(),
                                "assunto": msg.subject,
                                "remetente": msg.from_,
                                "data_recebimento": msg.date.isoformat() if msg.date else None,
                                "conteudo_preview": msg.text[:500] if msg.text else "Sem conteúdo texto",
                                "tem_anexos": len(msg.attachments) > 0,
                                "quantidade_anexos": len(msg.attachments)
                            }
                            batch_data.append(email_data)

                        except Exception as e:
                            logger.error(f"Erro ao extrair dados do email: {e}")
                            # Adicionar email com dados básicos
                            batch_data.append({
                                "timestamp": datetime.now().isoformat(),
                                "assunto": getattr(msg, 'subject', 'Erro ao obter assunto'),
                                "remetente": getattr(msg, 'from_', 'Erro ao obter remetente'),
                                "data_recebimento": None,
                                "conteudo_preview": "Erro ao extrair conteúdo",
                                "tem_anexos": False,
                                "quantidade_anexos": 0,
                                "erro_extracao": str(e)
                            })

                    # Classificar lote de emails
                    try:
                        classifications = self.classify_emails_batch(batch_data)

                        # Combinar dados com classificações
                        for i, email_data in enumerate(batch_data):
                            if i < len(classifications):
                                classification = classifications[i]
                            else:
                                # Classificação padrão se não houver correspondência
                                classification = {
                                    "email_numero": i + 1,
                                    "categoria": "ERRO_CORRESPONDENCIA",
                                    "urgencia": "MEDIA",
                                    "resumo": "Erro na correspondência de classificação",
                                    "acao_sugerida": "Revisar manualmente",
                                    "prazo_estimado": "N/A",
                                    "palavras_chave": ["erro"],
                                    "confianca": 0.0,
                                    "modelo_usado": self.model_config.model,
                                    "temperatura_usada": self.model_config.batch_temperature,
                                    "processamento_em_lote": True
                                }

                            email_processed = {
                                **email_data,
                                "classificacao": classification,
                                "lote_numero": batch_start // batch_size + 1,
                                "posicao_no_lote": i + 1
                            }

                            # Adicionar ao Dashboard Executivo se necessário
                            if self.notion_database_id:
                                self.add_to_executive_dashboard(email_data, classification)

                            all_processed_emails.append(email_processed)

                            logger.info(
                                f"✅ Email {batch_start + i + 1} processado: {classification['categoria']} - {classification['urgencia']}")

                    except Exception as e:
                        logger.error(f"Erro ao processar lote: {e}")
                        # Adicionar emails do lote com erro
                        for i, email_data in enumerate(batch_data):
                            email_processed = {
                                **email_data,
                                "erro_lote": str(e),
                                "classificacao": {
                                    "email_numero": i + 1,
                                    "categoria": "ERRO_LOTE",
                                    "urgencia": "MEDIA",
                                    "resumo": "Erro ao processar lote",
                                    "acao_sugerida": "Verificar logs",
                                    "prazo_estimado": "N/A",
                                    "palavras_chave": ["erro", "lote"],
                                    "confianca": 0.0,
                                    "modelo_usado": self.model_config.model,
                                    "temperatura_usada": self.model_config.batch_temperature,
                                    "processamento_em_lote": True
                                },
                                "lote_numero": batch_start // batch_size + 1,
                                "posicao_no_lote": i + 1
                            }
                            all_processed_emails.append(email_processed)

                # Salvar resultados em JSON
                self.save_results_json(all_processed_emails)

                logger.info(f"Total de emails processados: {len(all_processed_emails)}")
                logger.info(f"Processamento realizado em {(len(unread_emails) + batch_size - 1) // batch_size} lotes")

                return all_processed_emails

        except Exception as e:
            logger.error(f"Erro ao acessar emails: {e}")
            return []

    def save_results_json(self, processed_emails):
        """Salvar resultados em arquivo JSON com informações de lote"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emails_processados_lote_{timestamp}.json"

            results = {
                "timestamp_processamento": datetime.now().isoformat(),
                "configuracao_modelo": {
                    "modelo": self.model_config.model,
                    "temperatura_lote": self.model_config.batch_temperature,
                    "max_tokens": self.model_config.max_tokens,
                    "top_p": self.model_config.top_p,
                    "frequency_penalty": self.model_config.frequency_penalty,
                    "presence_penalty": self.model_config.presence_penalty,
                    "processamento_em_lote": True
                },
                "total_emails": len(processed_emails),
                "emails": processed_emails,
                "estatisticas": self.generate_statistics(processed_emails),
                "informacoes_lote": self.generate_batch_info(processed_emails)
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logger.info(f"Resultados salvos em: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Erro ao salvar JSON: {e}")
            return None

    def generate_batch_info(self, processed_emails):
        """Gerar informações sobre o processamento em lotes"""
        if not processed_emails:
            return {}

        batch_info = {
            "total_lotes": 0,
            "emails_por_lote": {},
            "chamadas_llm": 0
        }

        lotes_encontrados = set()

        for email in processed_emails:
            lote_num = email.get('lote_numero', 0)
            if lote_num > 0:
                lotes_encontrados.add(lote_num)

                if lote_num not in batch_info['emails_por_lote']:
                    batch_info['emails_por_lote'][lote_num] = 0
                batch_info['emails_por_lote'][lote_num] += 1

        batch_info['total_lotes'] = len(lotes_encontrados)
        batch_info['chamadas_llm'] = len(lotes_encontrados)  # Uma chamada por lote

        return batch_info

    def generate_statistics(self, processed_emails):
        """Gerar estatísticas dos emails processados"""
        if not processed_emails:
            return {}

        stats = {
            "por_categoria": {},
            "por_urgencia": {},
            "com_anexos": 0,
            "com_erro": 0,
            "confianca_media": 0.0,
            "confianca_por_categoria": {}
        }

        confidences = []

        for email in processed_emails:
            classificacao = email.get('classificacao', {})

            # Estatísticas por categoria
            categoria = classificacao.get('categoria', 'DESCONHECIDO')
            stats['por_categoria'][categoria] = stats['por_categoria'].get(categoria, 0) + 1

            # Estatísticas por urgência
            urgencia = classificacao.get('urgencia', 'DESCONHECIDO')
            stats['por_urgencia'][urgencia] = stats['por_urgencia'].get(urgencia, 0) + 1

            # Anexos
            if email.get('tem_anexos', False):
                stats['com_anexos'] += 1

            # Erros
            if ('erro' in email or 'erro_lote' in email or
                    categoria.startswith('ERRO')):
                stats['com_erro'] += 1

            # Confiança
            confianca = classificacao.get('confianca', 0.0)
            if isinstance(confianca, (int, float)) and confianca > 0:
                confidences.append(confianca)

                # Confiança por categoria
                if categoria not in stats['confianca_por_categoria']:
                    stats['confianca_por_categoria'][categoria] = []
                stats['confianca_por_categoria'][categoria].append(confianca)

        # Calcular confiança média
        if confidences:
            stats['confianca_media'] = sum(confidences) / len(confidences)

            # Confiança média por categoria
            for categoria, conf_list in stats['confianca_por_categoria'].items():
                stats['confianca_por_categoria'][categoria] = sum(conf_list) / len(conf_list)

        return stats

    def test_connections(self):
        """Testar todas as conexões necessárias"""
        results = {}

        # Testar OpenAI
        try:
            params = self.model_config.get_test_params()
            response = self.openai_client.chat.completions.create(
                messages=[{"role": "user", "content": "Teste de conexão - responda apenas 'OK'"}],
                **params
            )
            results['openai'] = f"✅ Conectado ({params['model']})"
            logger.info("Conexão OpenAI: OK")
        except Exception as e:
            results['openai'] = f"❌ Erro: {str(e)[:100]}"
            logger.error(f"Conexão OpenAI: ERRO - {e}")

        # Testar Gmail
        try:
            with MailBox('imap.gmail.com').login(self.gmail_user, self.gmail_password) as mailbox:
                mailbox.folder.set('INBOX')
                results['gmail'] = f"✅ Conectado"
                logger.info("Conexão Gmail: OK")
        except Exception as e:
            results['gmail'] = f"❌ Erro: {str(e)[:100]}"
            logger.error(f"Conexão Gmail: ERRO - {e}")

        return results

    def print_summary(self, processed_emails):
        """Imprimir resumo dos resultados com informações de lote"""
        if not processed_emails:
            print("\n📭 Nenhum email foi processado.")
            return

        print(f"\n📊 RESUMO DO PROCESSAMENTO EM LOTE")
        print(f"{'=' * 50}")
        print(f"Total de emails processados: {len(processed_emails)}")

        # Informações de lote
        batch_info = self.generate_batch_info(processed_emails)
        if batch_info['total_lotes'] > 0:
            print(f"🔄 Processamento em lotes:")
            print(f"  • Total de lotes: {batch_info['total_lotes']}")
            print(f"  • Chamadas ao LLM: {batch_info['chamadas_llm']}")
            print(f"  • Economia estimada: {len(processed_emails) - batch_info['chamadas_llm']} chamadas")

        # Estatísticas
        stats = self.generate_statistics(processed_emails)

        print(f"\n📈 Por categoria:")
        for categoria, count in stats['por_categoria'].items():
            confianca_cat = stats['confianca_por_categoria'].get(categoria, 0)
            if confianca_cat > 0:
                print(f"  • {categoria}: {count} (confiança média: {confianca_cat:.2f})")
            else:
                print(f"  • {categoria}: {count}")

        print(f"\n⚡ Por urgência:")
        for urgencia, count in stats['por_urgencia'].items():
            print(f"  • {urgencia}: {count}")

        print(f"\n📎 Com anexos: {stats['com_anexos']}")
        print(f"❌ Com erro: {stats['com_erro']}")

        if stats['confianca_media'] > 0:
            print(f"🎯 Confiança média: {stats['confianca_media']:.2f}")

        # Mostrar emails urgentes
        urgent_emails = [e for e in processed_emails
                         if e.get('classificacao', {}).get('urgencia') == 'ALTA']

        if urgent_emails:
            print(f"\n🚨 EMAILS URGENTES ({len(urgent_emails)}):")
            for email in urgent_emails:
                classificacao = email['classificacao']
                print(f"  • {email['assunto'][:60]}...")
                print(f"    De: {email['remetente']}")
                print(f"    Ação: {classificacao['acao_sugerida']}")
                if 'confianca' in classificacao and classificacao['confianca'] > 0:
                    print(f"    Confiança: {classificacao['confianca']:.2f}")
                print()


def create_batch_config():
    """Criar configuração otimizada para processamento em lote"""
    config = ModelConfig()

    # Configuração otimizada para lotes
    config.batch_temperature = 0.15  # Baixa para consistência
    config.max_tokens = 2000  # Maior para múltiplos emails
    config.top_p = 0.9  # Balanceado

    return config


def main():
    """Função principal"""
    try:
        print("🤖 AGENTE UFRN - Análise Inteligente de Emails (PROCESSAMENTO EM LOTE)")
        print("=" * 70)

        # Criar configuração otimizada para lotes
        model_config = create_batch_config()

        # Inicializar agente com configuração
        agent = AgentUFRN(model_config)

        # Testar conexões
        print("\n🔍 Testando conexões...")
        test_results = agent.test_connections()

        for service, status in test_results.items():
            print(f"  {service.upper()}: {status}")

        # Verificar se pode prosseguir
        if not all("✅" in status for status in test_results.values()):
            print("\n❌ Não é possível prosseguir devido a problemas de conexão.")
            print("Verifique suas credenciais no arquivo .env")
            return

        # Processar emails em lotes
        print("\n📧 Processando emails não lidos em lotes...")
        print("💡 Vantagem: Múltiplos emails processados por chamada ao LLM")

        processed_emails = agent.process_emails(limit=10, batch_size=2)  # 2 emails por lote

        # Mostrar resumo
        agent.print_summary(processed_emails)

        print(f"\n✅ Processamento em lote concluído!")
        print(f"📄 Resultados detalhados salvos em arquivo JSON")
        print(f"💰 Economia de custos: Menos chamadas à API OpenAI")

    except Exception as e:
        logger.error(f"Erro na execução principal: {e}")
        print(f"\n❌ Erro: {e}")
        print("Verifique o arquivo 'agent_ufrn.log' para mais detalhes.")


if __name__ == "__main__":
    main()

