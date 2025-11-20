# generate_diagram.py
import logging
from agents.graph import MASGraphExecutor

# Configura um logging básico para vermos as mensagens de sucesso
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    """
    Este script gera o diagrama do fluxo do MAS (LangGraph).
    """
    print("Gerando diagrama do MAS...")
    
    try:
        # 1. Instancia o seu executor (que constrói o grafo)
        #   
        executor = MASGraphExecutor()
        
        # 2. Chama o método 'visualize' que já existe na sua classe
        #   
        output_filename = "fluxo_agentes.png"
        executor.visualize(output_filename)
        
        print(f"\nDiagrama salvo com sucesso em: {output_filename}")

    except ImportError as e:
        print(f"\n[ERRO] Falha na importação: {e}")
        print("Certifique-se de que está executando este script a partir do diretório raiz do projeto.")
    except Exception as e:
        print(f"\n[ERRO] Falha ao gerar o diagrama: {e}")
        print("Verifique se as dependências do Passo 1 (graphviz, pygraphviz) foram instaladas corretamente.")

if __name__ == "__main__":
    main()
