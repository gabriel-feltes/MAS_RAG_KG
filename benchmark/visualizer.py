#!/usr/bin/env python3
"""
Visualização e Análise de Resultados de Benchmark do Sistema Multi-Agent RAG.

Este script gera visualizações abrangentes a partir do JSON de resultados de benchmark,
incluindo comparações de desempenho entre tipos de busca, categorias e níveis de complexidade.

Uso:
    python plotar_resultados.py

O script espera por 'results2.json' no mesmo diretório.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import warnings

warnings.filterwarnings("ignore")

# Configuração
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
TAMANHO_FIGURA = (16, 10)
RESOLUCAO = 300


class VisualizadorBenchmarkRAG:
    """Visualiza resultados de benchmark RAG a partir de saída JSON."""

    def __init__(self, arquivo_json: str = "results2.json"):
        """Carrega e analisa resultados de benchmark."""
        self.caminho_arquivo = Path(arquivo_json)

        if not self.caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_json}")

        with open(self.caminho_arquivo, "r") as f:
            self.dados = json.load(f)

        # Converte para DataFrame do pandas para análise mais fácil
        self.df = pd.DataFrame(self.dados["query_results"])

        print(f"✅ Carregados {len(self.df)} resultados de benchmark")
        print(f"   Métricas: {len(self.dados['search_type_metrics'])} tipos de busca")
        print(f"   Categorias: {len(self.dados['category_metrics'])} categorias")
        print(f"   Duração: {self.dados['duration_seconds']:.1f} segundos\n")

    def plotar_comparacao_tipos_busca(self):
        """Compara desempenho entre tipos de busca (Vetor, Grafo, Híbrido)."""
        fig, axes = plt.subplots(2, 2, figsize=TAMANHO_FIGURA)
        fig.suptitle("Comparação de Desempenho por Tipo de Busca", fontsize=18, fontweight="bold", y=1.00)

        tipos_busca = ["vector", "graph", "hybrid"]
        cores = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # 1. Pontuação de Alucinação
        alucinacao = [self.dados["search_type_metrics"][st]["avg_hallucination"] for st in tipos_busca]
        axes[0, 0].bar(tipos_busca, alucinacao, color=cores, alpha=0.8, edgecolor="black", linewidth=2)
        axes[0, 0].set_ylabel("Pontuação de Alucinação", fontsize=12, fontweight="bold")
        axes[0, 0].set_title("Pontuação Média de Alucinação", fontsize=13, fontweight="bold")
        axes[0, 0].set_ylim(0, max(alucinacao) * 1.2)
        for i, v in enumerate(alucinacao):
            axes[0, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

        # 2. Latência
        latencia = [self.dados["search_type_metrics"][st]["avg_latency_ms"] for st in tipos_busca]
        axes[0, 1].bar(tipos_busca, latencia, color=cores, alpha=0.8, edgecolor="black", linewidth=2)
        axes[0, 1].set_ylabel("Latência (ms)", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("Latência Total Média", fontsize=13, fontweight="bold")
        for i, v in enumerate(latencia):
            axes[0, 1].text(i, v + 500, f"{v:.0f}ms", ha="center", fontweight="bold")

        # 3. Acurácia de Citação
        citacao = [self.dados["search_type_metrics"][st]["avg_citation_accuracy"] for st in tipos_busca]
        axes[1, 0].bar(tipos_busca, citacao, color=cores, alpha=0.8, edgecolor="black", linewidth=2)
        axes[1, 0].set_ylabel("Acurácia de Citação", fontsize=12, fontweight="bold")
        axes[1, 0].set_title("Acurácia Média de Citação", fontsize=13, fontweight="bold")
        axes[1, 0].set_ylim(0.8, 1.05)
        for i, v in enumerate(citacao):
            axes[1, 0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

        # 4. Precisão de Recuperação
        precisao = [self.dados["search_type_metrics"][st]["avg_retrieval_precision"] for st in tipos_busca]
        axes[1, 1].bar(tipos_busca, precisao, color=cores, alpha=0.8, edgecolor="black", linewidth=2)
        axes[1, 1].set_ylabel("Precisão de Recuperação", fontsize=12, fontweight="bold")
        axes[1, 1].set_title("Precisão Média de Recuperação", fontsize=13, fontweight="bold")
        axes[1, 1].set_ylim(0, max(precisao) * 1.2 if max(precisao) > 0 else 0.5)
        for i, v in enumerate(precisao):
            axes[1, 1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig("01_comparacao_tipos_busca.png", dpi=RESOLUCAO, bbox_inches="tight")
        print("✅ Salvo: 01_comparacao_tipos_busca.png")
        plt.close()

    def plotar_desempenho_categoria(self):
        """Comparação de desempenho entre categorias de query."""
        fig, axes = plt.subplots(2, 2, figsize=TAMANHO_FIGURA)
        fig.suptitle("Desempenho por Categoria de Query", fontsize=18, fontweight="bold", y=1.00)

        categorias = list(self.dados["category_metrics"].keys())

        metricas = {
            "Alucinação": [self.dados["category_metrics"][cat]["avg_hallucination"] for cat in categorias],
            "Acurácia de Citação": [self.dados["category_metrics"][cat]["avg_citation_accuracy"] for cat in categorias],
            "Latência (ms)": [self.dados["category_metrics"][cat]["avg_latency_ms"] for cat in categorias],
            "Contagem": [self.dados["category_metrics"][cat]["count"] for cat in categorias]
        }

        for idx, (nome_metrica, valores) in enumerate(metricas.items()):
            ax = axes[idx // 2, idx % 2]
            barras = ax.bar(categorias, valores, alpha=0.8, edgecolor="black", linewidth=2)

            # Gradiente de cores
            cores_gradiente = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(categorias)))
            for barra, cor in zip(barras, cores_gradiente):
                barra.set_color(cor)

            ax.set_ylabel(nome_metrica, fontsize=12, fontweight="bold")
            ax.set_title(f"{nome_metrica} por Categoria", fontsize=13, fontweight="bold")
            ax.tick_params(axis="x", rotation=45)

            for i, v in enumerate(valores):
                ax.text(i, v + (max(valores) * 0.02), f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)

        plt.tight_layout()
        plt.savefig("02_desempenho_categoria.png", dpi=RESOLUCAO, bbox_inches="tight")
        print("✅ Salvo: 02_desempenho_categoria.png")
        plt.close()

    def plotar_degradacao_complexidade(self):
        """Mostra como o desempenho degrada com a complexidade da query."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Desempenho vs Complexidade da Query", fontsize=18, fontweight="bold", y=1.05)

        niveis_complexidade = ["simple", "moderate", "complex"]

        # Alucinação por complexidade
        alucinacao = [self.dados["complexity_metrics"][nivel]["avg_hallucination"] for nivel in niveis_complexidade]
        axes[0].plot(niveis_complexidade, alucinacao, marker="o", linewidth=3, markersize=12, color="#FF6B6B")
        axes[0].fill_between(range(len(niveis_complexidade)), alucinacao, alpha=0.3, color="#FF6B6B")
        axes[0].set_ylabel("Pontuação de Alucinação", fontsize=12, fontweight="bold")
        axes[0].set_title("Degradação de Alucinação com Complexidade", fontsize=13, fontweight="bold")
        axes[0].grid(True, alpha=0.3)
        # Ajusta a posição do texto para não sobrepor o título
        for i, v in enumerate(alucinacao):
            axes[0].text(i, v + 0.03 * (max(alucinacao) if max(alucinacao) > 0 else 1), f"{v:.3f}", ha="center", fontweight="bold")

        # Latência por complexidade
        latencia = [self.dados["complexity_metrics"][nivel]["avg_latency_ms"] for nivel in niveis_complexidade]
        axes[1].plot(niveis_complexidade, latencia, marker="s", linewidth=3, markersize=12, color="#45B7D1")
        axes[1].fill_between(range(len(niveis_complexidade)), latencia, alpha=0.3, color="#45B7D1")
        axes[1].set_ylabel("Latência (ms)", fontsize=12, fontweight="bold")
        axes[1].set_title("Aumento de Latência com Complexidade", fontsize=13, fontweight="bold")
        axes[1].grid(True, alpha=0.3)
        for i, v in enumerate(latencia):
            axes[1].text(i, v + 500, f"{v:.0f}ms", ha="center", fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("03_degradacao_complexidade.png", dpi=RESOLUCAO, bbox_inches="tight")
        print("✅ Salvo: 03_degradacao_complexidade.png")
        plt.close()

    def plotar_decomposicao_latencia(self):
        """Decomposição de latência por fase: recuperação, síntese, validação."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Decomposição de Latência por Fase", fontsize=18, fontweight="bold", y=1.00)

        tipos_busca = ["vector", "graph", "hybrid"]

        # Obtém latências médias para cada fase
        tempos_recuperacao = []
        tempos_sintese = []
        tempos_validacao = []

        for st in tipos_busca:
            df_st = self.df[self.df["search_type"] == st]
            tempos_recuperacao.append(df_st["retrieval_latency_ms"].mean())
            tempos_sintese.append(df_st["synthesis_latency_ms"].mean())
            tempos_validacao.append(df_st["validation_latency_ms"].mean())

        # Gráfico de barras empilhadas
        x = np.arange(len(tipos_busca))
        largura = 0.6

        p1 = axes[0].bar(x, tempos_recuperacao, largura, label="Recuperação", color="#FF6B6B", edgecolor="black", linewidth=2)
        p2 = axes[0].bar(x, tempos_sintese, largura, bottom=tempos_recuperacao, label="Síntese", color="#4ECDC4", edgecolor="black", linewidth=2)
        p3 = axes[0].bar(x, tempos_validacao, largura, bottom=np.array(tempos_recuperacao) + np.array(tempos_sintese),
                         label="Validação", color="#45B7D1", edgecolor="black", linewidth=2)

        axes[0].set_ylabel("Latência (ms)", fontsize=12, fontweight="bold")
        axes[0].set_title("Latência Empilhada por Fase", fontsize=13, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tipos_busca)
        axes[0].legend(loc="upper left", fontsize=11, framealpha=0.9)
        axes[0].grid(True, alpha=0.3, axis="y")

        # Gráfico de pizza para média total
        total_por_fase = {
            "Recuperação": np.mean(tempos_recuperacao),
            "Síntese": np.mean(tempos_sintese),
            "Validação": np.mean(tempos_validacao)
        }
        cores_pizza = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        wedges, texts, autotexts = axes[1].pie(total_por_fase.values(), labels=total_por_fase.keys(),
                                                autopct="%1.1f%%", colors=cores_pizza, startangle=90,
                                                textprops={"fontsize": 12, "fontweight": "bold"})
        axes[1].set_title("Distribuição de Tempo Médio entre Todas as Queries", fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.savefig("04_decomposicao_latencia.png", dpi=RESOLUCAO, bbox_inches="tight")
        print("✅ Salvo: 04_decomposicao_latencia.png")
        plt.close()

    def plotar_alucinacao_vs_latencia(self):
        """Gráfico de dispersão: alucinação vs latência para cada query."""
        fig, ax = plt.subplots(figsize=(12, 8))

        tipos_busca = self.df["search_type"].unique()
        dicionario_cores = {"vector": "#FF0000", "graph": "#66FF00", "hybrid": "#0044FF"}

        for st in tipos_busca:
            df_st = self.df[self.df["search_type"] == st]
            ax.scatter(df_st["total_latency_ms"], df_st["hallucination_score"],
                      label=st.capitalize(), s=150, alpha=0.6, color=dicionario_cores[st], edgecolors="black", linewidth=1.5)

        ax.set_xlabel("Latência (ms)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pontuação de Alucinação", fontsize=12, fontweight="bold")
        ax.set_title("Trade-off: Alucinação vs Latência", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("05_alucinacao_vs_latencia.png", dpi=RESOLUCAO, bbox_inches="tight")
        print("✅ Salvo: 05_alucinacao_vs_latencia.png")
        plt.close()

    def plotar_resumo_sucesso(self):
        """Taxa de sucesso geral e estatísticas."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Resumo Geral de Benchmark", fontsize=18, fontweight="bold", y=0.98)

        # 1. Taxa de sucesso
        total = self.dados["total_queries"]
        completadas = self.dados["completed_queries"]
        falhadas = self.dados["failed_queries"]

        axes[0, 0].bar(["Completadas", "Falhadas"], [completadas, falhadas], color=["#2ECC71", "#E74C3C"], alpha=0.8, edgecolor="black", linewidth=2)
        axes[0, 0].set_ylabel("Contagem", fontsize=12, fontweight="bold")
        axes[0, 0].set_title(f"Conclusão de Query (Taxa de Sucesso: {completadas/total*100:.1f}%)", fontsize=13, fontweight="bold")
        axes[0, 0].set_ylim(0, total * 1.1)

        # 2. Métricas médias
        nomes_metricas = ["Alucinação", "Acurácia de Citação", "Confiança"]
        valores_metricas = [self.dados["avg_hallucination_score"], self.dados["avg_citation_accuracy"], self.dados["avg_answer_confidence"]]
        barras = axes[0, 1].barh(nomes_metricas, valores_metricas, color=["#FF6B6B", "#2ECC71", "#3498DB"], alpha=0.8, edgecolor="black", linewidth=2)
        axes[0, 1].set_xlabel("Pontuação", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("Métricas Médias entre Todas as Queries", fontsize=13, fontweight="bold")
        axes[0, 1].set_xlim(0, 1.1)
        for i, v in enumerate(valores_metricas):
            axes[0, 1].text(v + 0.02, i, f"{v:.3f}", va="center", fontweight="bold")

        # 3. Queries por categoria
        categorias = list(self.dados["category_metrics"].keys())
        contagens = [self.dados["category_metrics"][cat]["count"] for cat in categorias]
        cores_cat = plt.cm.Set3(np.linspace(0, 1, len(categorias)))
        axes[1, 0].pie(contagens, labels=categorias, autopct="%1.1f%%", colors=cores_cat, startangle=90, textprops={"fontsize": 11, "fontweight": "bold"})
        axes[1, 0].set_title("Queries por Categoria", fontsize=13, fontweight="bold")

        # 4. Distribuição de tipo de busca
        tipos_busca = list(self.dados["search_type_metrics"].keys())
        contagens_st = [self.dados["search_type_metrics"][st]["count"] for st in tipos_busca]
        cores_st = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        axes[1, 1].pie(contagens_st, labels=[s.capitalize() for s in tipos_busca], autopct="%1.1f%%", colors=cores_st, startangle=90, textprops={"fontsize": 11, "fontweight": "bold"})
        axes[1, 1].set_title("Queries por Tipo de Busca", fontsize=13, fontweight="bold")

        plt.tight_layout()
        plt.savefig("06_resumo_estatisticas.png", dpi=RESOLUCAO, bbox_inches="tight")
        print("✅ Salvo: 06_resumo_estatisticas.png")
        plt.close()

    def gerar_todos_graficos(self):
        """Gera todos os gráficos de visualização."""
        print("\n" + "="*70)
        print("📊 GERANDO VISUALIZAÇÕES")
        print("="*70 + "\n")

        self.plotar_comparacao_tipos_busca()
        self.plotar_desempenho_categoria()
        self.plotar_degradacao_complexidade()
        self.plotar_decomposicao_latencia()
        self.plotar_alucinacao_vs_latencia()
        self.plotar_resumo_sucesso()

        print("\n" + "="*70)
        print("✅ TODAS AS VISUALIZAÇÕES COMPLETAS!")
        print("="*70)
        print("\nArquivos gerados:")
        print("  1. 01_comparacao_tipos_busca.png")
        print("  2. 02_desempenho_categoria.png")
        print("  3. 03_degradacao_complexidade.png")
        print("  4. 04_decomposicao_latencia.png")
        print("  5. 05_alucinacao_vs_latencia.png")
        print("  6. 06_resumo_estatisticas.png\n")


def principal():
    """Ponto de entrada principal."""
    try:
        visualizador = VisualizadorBenchmarkRAG("results2.json")
        visualizador.gerar_todos_graficos()
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    principal()