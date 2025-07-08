# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="HpcEPA45sNXT"
# ==============================================================================
# 1. SETUP - IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import re
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Layout

# ==============================================================================
# 2. DEFINIÇÃO DA FUNÇÃO DE PROCESSAMENTO (VERSÃO FINAL 6.0 - COM DEBUG)
# ==============================================================================

def final_parse_and_process_flash_data(file_content, diametro_mm, espessura_inicial_mm, espessura_final_mm, debug_mode=False):
    """
    Versão Final 6.0: Inclui um modo de depuração (debug_mode) para imprimir
    logs detalhados do processo de parsing e validação.
    """
    log_messages = []
    def log(message):
        if debug_mode:
            log_messages.append(message)

    try:
        # --- ETAPA 1: PARSER INTELIGENTE DO ARQUIVO TXT ---
        content_str = file_content.decode('latin-1')
        lines = content_str.splitlines()
        log(f"Arquivo lido com sucesso. Total de {len(lines)} linhas.")

        header_line_index = -1
        header_line = ""
        keywords = ['time', 'tempo', 'tensao', 'voltage', 'corrente', 'current', 'retra', 'temp']
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if sum(keyword in line_lower for keyword in keywords) >= 3:
                header_line_index = i
                header_line = line
                log(f"Linha de cabeçalho detectada na linha {i}: '{header_line}'")
                break

        if header_line_index == -1:
            raise ValueError("Não foi possível encontrar a linha de cabeçalho. Verifique o formato do TXT.")

        data_content = "\n".join(lines[header_line_index + 1:])

        df_raw = pd.read_csv(
            io.StringIO(data_content), sep=r'\s+', decimal=',', header=None, engine='python', dtype=str
        )
        log("Dados brutos carregados em um DataFrame. Primeiras linhas:")
        log(df_raw.head().to_string())

        raw_columns = re.split(r'\s+', header_line.strip())
        num_cols = min(len(raw_columns), df_raw.shape[1])
        df_raw = df_raw.iloc[:, :num_cols]
        df_raw.columns = raw_columns[:num_cols]
        log(f"Colunas brutas identificadas: {list(df_raw.columns)}")

        # --- ETAPA 2: MAPEAMENTO DE COLUNAS ---
        column_map = {
            'Tempo': ['Tempo', 'Time'],
            'Tensao': ['Tensao', 'Tensão', 'Voltage'],
            'Corrente': ['Corrente', 'Current'],
            'Retracao': ['Retracao', 'Retração', 'Shrinkage'],
            'Temp_Amostra': ['TempAmostra', 'Temp. Amostra'],
            'Temp_Forno': ['TempForno', 'Temp. Forno'],
            'Horario': ['Horario']
        }

        rename_dict = {}
        for col_name in df_raw.columns:
            for standard_name, possible_names in column_map.items():
                for possible_name in possible_names:
                    if possible_name.lower() in col_name.lower():
                        rename_dict[col_name] = standard_name
                        break
        df_raw.rename(columns=rename_dict, inplace=True)
        log(f"Colunas após renomeação padrão: {list(df_raw.columns)}")

        # --- ETAPA 3: CONVERSÃO E VALIDAÇÃO DE DADOS ---
        required_cols = ['Tempo', 'Tensao', 'Corrente', 'Retracao', 'Temp_Amostra', 'Temp_Forno']
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            log(f"ERRO DE VALIDAÇÃO: Colunas essenciais faltando: {missing_cols}")
            raise ValueError(f"Não foi possível encontrar as seguintes colunas essenciais: {', '.join(missing_cols)}. Verifique o cabeçalho do arquivo TXT ou o log de depuração.")

        for col in required_cols:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].str.replace(',', '.', regex=False)
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        df_raw.dropna(subset=['Tempo', 'Tensao', 'Corrente'], inplace=True)
        log("Colunas convertidas para formato numérico com sucesso.")

        # --- ETAPA 4: CÁLCULOS FÍSICOS ---
        df = df_raw.copy()
        if df['Tempo'].iloc[0] > 10000: # Heurística para detectar milissegundos
             df['Tempo'] = df['Tempo'] / 1000.0

        area_cm2 = np.pi * ((diametro_mm / 2 / 10) ** 2)
        retracao_global_mm = espessura_inicial_mm - espessura_final_mm

        if 'Corrente' in df.columns and df['Corrente'].max() > 100:
             df['Corrente_A'] = df['Corrente'] / 1000.0
        else:
             df['Corrente_A'] = df['Corrente']

        df['Potencia_W'] = df['Tensao'] * df['Corrente_A']
        df['Densidade_Potencia_W_cm3'] = df['Potencia_W'] / (area_cm2 * (espessura_inicial_mm / 10))
        df['Densidade_Corrente_A_cm2'] = df['Corrente_A'] / area_cm2
        df['Retracao_Corrigida'] = df['Retracao'] - df['Retracao'].iloc[0]

        sigma, emissividade = 5.670374e-8, 0.9
        area_m2 = area_cm2 / 10000
        potencia_nao_nula = df['Potencia_W'].clip(lower=1e-9)
        df['Temp_BBR_K'] = (potencia_nao_nula / (emissividade * sigma * area_m2)) ** 0.25
        df['Temp_BBR_C'] = df['Temp_BBR_K'] - 273.15
        log("Cálculos físicos principais concluídos.")

        # --- ETAPA 5: FAIXA DE INTERESSE, NORMALIZAÇÃO DO TEMPO E AJUSTE 4 ---
        corrente_maxima = df['Corrente_A'].max()
        limiar_corrente = corrente_maxima * 0.10
        idx_inicio_flash = df[df['Corrente_A'] >= limiar_corrente].index[0]
        idx_corrente_maxima = df['Corrente_A'].idxmax()
        df_flash = df.loc[idx_inicio_flash:idx_corrente_maxima].copy()

        tempo_inicial_flash = df_flash['Tempo'].iloc[0]
        df_flash['Tempo_Normalizado'] = df_flash['Tempo'] - tempo_inicial_flash
        log("Faixa de interesse identificada e tempo normalizado.")

        retracao_no_inicio_flash = df_flash['Retracao_Corrigida'].iloc[0]
        retracao_no_fim_flash = df_flash['Retracao_Corrigida'].iloc[-1]
        delta_retracao_medida = retracao_no_fim_flash - retracao_no_inicio_flash
        df_flash['Retracao_Ajuste4'] = 0 if delta_retracao_medida == 0 else ((df_flash['Retracao_Corrigida'] - retracao_no_inicio_flash) / delta_retracao_medida) * retracao_global_mm

        df_flash['Taxa_Retracao_vs_Tempo'] = df_flash['Retracao_Ajuste4'].diff() / df_flash['Tempo_Normalizado'].diff()
        df_flash['Taxa_Retracao_vs_Temp'] = df_flash['Retracao_Ajuste4'].diff() / df_flash['Temp_Amostra'].diff()
        df_flash.fillna(0, inplace=True)
        log("Cálculos finais e derivadas concluídos.")

        return df_flash, "\n".join(log_messages)
    except Exception as e:
        log(f"!!! EXCEÇÃO CAPTURADA: {e} !!!")
        return None, "\n".join(log_messages)

# ==============================================================================
# 3. CRIAÇÃO DA INTERFACE GRÁFICA (GUI com ipywidgets)
# ==============================================================================

button_layout = Layout(width='200px', height='40px')
uploader = widgets.FileUpload(accept='.txt', description='1. Carregar Arquivo TXT', style={'description_width': 'initial'}, layout=Layout(width='350px'))
diametro_input = widgets.FloatText(description="Diâmetro (mm):", style={'description_width': 'initial'})
esp_inicial_input = widgets.FloatText(description="Espessura Inicial (mm):", style={'description_width': 'initial'})
esp_final_input = widgets.FloatText(description="Espessura Final (mm):", style={'description_width': 'initial'})
debug_checkbox = widgets.Checkbox(value=False, description='Ativar Modo de Depuração', indent=False)

process_button = widgets.Button(description="2. Processar Dados", button_style='success', layout=button_layout)
basic_graphs_button = widgets.Button(description="Gráficos Básicos", button_style='info', layout=button_layout)
adv_graphs_button = widgets.Button(description="Gráficos Adicionais", button_style='info', layout=button_layout)
export_button = widgets.Button(description="3. Exportar para CSV", button_style='primary', layout=button_layout)

status_output = widgets.Output()
graphs_output = widgets.Output()
export_output = widgets.Output()

processed_df = None

def on_process_button_clicked(b):
    global processed_df
    status_output.clear_output(); graphs_output.clear_output(); export_output.clear_output()
    with status_output:
        if not uploader.value: print("Erro: Nenhum arquivo foi carregado."); return
        file_info = list(uploader.value.values())[0]
        diametro, esp_inicial, esp_final = diametro_input.value, esp_inicial_input.value, esp_final_input.value
        if not all([diametro, esp_inicial, esp_final]): print("Erro: Todos os campos de dimensão devem ser preenchidos."); return

        df, log_output = final_parse_and_process_flash_data(
            file_info['content'], diametro, esp_inicial, esp_final, debug_mode=debug_checkbox.value
        )

        if debug_checkbox.value:
            print("--- LOG DE DEPURAÇÃO ---")
            print(log_output)
            print("----------------------")

        if df is None:
            print("\nOcorreu um erro durante o processamento. Verifique o log de depuração acima para detalhes.")
        else:
            processed_df = df
            print(f"\n✔ Processamento concluído! {len(df)} pontos na faixa de interesse.")

def create_download_link(df):
    original_filename = list(uploader.value.keys())[0].replace('.txt', '')
    download_filename = f"Resultados_Auto_{original_filename}.csv"
    df_to_export = df.copy()
    df_to_export.rename(columns={'Tempo_Normalizado': 'Tempo_Flash (s)'}, inplace=True)
    csv_str = df_to_export.to_csv(index=False, decimal=',', sep=';')
    b64 = base64.b64encode(csv_str.encode('utf-8-sig')).decode()
    link = f'<a href="data:text/csv;base64,{b64}" download="{download_filename}">Clique aqui para baixar o arquivo CSV: {download_filename}</a>'
    display(HTML(link))

def on_basic_graphs_clicked(b):
    with graphs_output:
        graphs_output.clear_output()
        if processed_df is None: print("Por favor, processe os dados primeiro."); return
        fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8)); fig1.suptitle('Análise de Flash Sintering - Painel Principal', fontsize=16)
        t = processed_df['Tempo_Normalizado']
        xlabel = 'Tempo Normalizado (s)'
        axs1[0, 0].plot(t, processed_df['Densidade_Potencia_W_cm3'], 'r'); axs1[0, 0].set(title='Densidade de Potência vs. Tempo', xlabel=xlabel, ylabel='W/cm³'); axs1[0, 0].grid(True)
        axs1[0, 1].plot(t, processed_df['Temp_Amostra'], 'g'); axs1[0, 1].set(title='Temperatura vs. Tempo', xlabel=xlabel, ylabel='°C'); axs1[0, 1].grid(True)
        axs1[1, 0].plot(t, processed_df['Densidade_Corrente_A_cm2'], 'b'); axs1[1, 0].set(title='Densidade de Corrente vs. Tempo', xlabel=xlabel, ylabel='A/cm²'); axs1[1, 0].grid(True)
        axs1[1, 1].plot(t, processed_df['Retracao_Ajuste4'], 'm'); axs1[1, 1].set(title='Retração Ajustada vs. Tempo', xlabel=xlabel, ylabel='Retração (mm)'); axs1[1, 1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def on_adv_graphs_clicked(b):
    with graphs_output:
        graphs_output.clear_output()
        if processed_df is None: print("Por favor, processe os dados primeiro."); return
        fig2, axs2 = plt.subplots(2, 2, figsize=(12, 8)); fig2.suptitle('Análise de Flash Sintering - Painel Adicional', fontsize=16)
        t = processed_df['Tempo_Normalizado']
        xlabel = 'Tempo Normalizado (s)'
        axs2[0, 0].plot(t, processed_df['Taxa_Retracao_vs_Tempo'], 'c'); axs2[0, 0].set(title='Taxa de Retração vs. Tempo', xlabel=xlabel, ylabel='mm/s'); axs2[0, 0].grid(True)
        axs2[0, 1].plot(processed_df['Temp_Amostra'], processed_df['Taxa_Retracao_vs_Temp'], 'y'); axs2[0, 1].set(title='Taxa de Retração vs. Temperatura', xlabel='Temperatura (°C)', ylabel='mm/°C'); axs2[0, 1].grid(True)
        axs2[1, 0].plot(t, processed_df['Temp_BBR_C'], 'orange'); axs2[1, 0].set(title='Temperatura BBR vs. Tempo', xlabel=xlabel, ylabel='°C'); axs2[1, 0].grid(True)
        axs2[1, 1].plot(processed_df['Temp_BBR_C'], processed_df['Retracao_Ajuste4'], 'purple'); axs2[1, 1].set(title='Retração vs. Temperatura BBR', xlabel='BBR (°C)', ylabel='Retração (mm)'); axs2[1, 1].grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def on_export_button_clicked(b):
    with export_output:
        export_output.clear_output()
        if processed_df is None: print("Por favor, processe os dados primeiro."); return
        create_download_link(processed_df)

process_button.on_click(on_process_button_clicked)
basic_graphs_button.on_click(on_basic_graphs_clicked)
adv_graphs_button.on_click(on_adv_graphs_clicked)
export_button.on_click(on_export_button_clicked)

# ==============================================================================
# 4. MONTAGEM E EXIBIÇÃO DA INTERFACE
# ==============================================================================
input_box = widgets.VBox([uploader, widgets.HBox([diametro_input, esp_inicial_input, esp_final_input]), debug_checkbox])
action_box = widgets.VBox([process_button, export_button])
graphs_buttons_box = widgets.HBox([basic_graphs_button, adv_graphs_button])
title = widgets.HTML("<h3>Aplicação Final para Análise de Dados de Flash Sintering (v6.0 - com Debug)</h3>")

display(widgets.VBox([title, widgets.HBox([input_box, action_box]), status_output, graphs_buttons_box, graphs_output, export_output]))
