import pandas as pd
from scipy import signal


def aplicar_filtro_butterworth(
    df: pd.DataFrame,
    coluna_sinal: str,
    freq_corte_hz: float,
    freq_amostragem_hz: float,
    ordem_filtro: int = 4,
) -> pd.Series:
    """Aplica um filtro Butterworth passa-baixa de fase nula a uma coluna de sinal.

    Args:
        df: DataFrame contendo os dados brutos.
        coluna_sinal: Nome da coluna do DataFrame que armazena o sinal a ser filtrado.
        freq_corte_hz: Frequência de corte em hertz do filtro passa-baixa.
        freq_amostragem_hz: Frequência de amostragem do sinal em hertz.
        ordem_filtro: Ordem do filtro Butterworth (padronizado como 4).

    Returns:
        pandas.Series com o sinal filtrado em fase nula.

    Raises:
        ValueError: Se ``freq_corte_hz`` não estiver entre 0 e ``freq_amostragem_hz / 2``.
    """
    if not 0 < freq_corte_hz < (freq_amostragem_hz / 2):
        raise ValueError("freq_corte_hz deve ser positiva e menor que a frequência de Nyquist")

    # Normalização da frequência de corte utilizando a frequência de Nyquist
    freq_nyquist = freq_amostragem_hz / 2
    freq_normalizada = freq_corte_hz / freq_nyquist

    b, a = signal.butter(ordem_filtro, freq_normalizada, btype="low", analog=False)

    sinal = df[coluna_sinal].to_numpy()

    sinal_filtrado = signal.filtfilt(b, a, sinal)

    return pd.Series(sinal_filtrado, index=df.index)
