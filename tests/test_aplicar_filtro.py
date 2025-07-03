import pandas as pd
import numpy as np
import pytest

from aplicacao_filtro import aplicar_filtro_butterworth


def test_aplicar_filtro_butterworth_invalid_cutoff():
    df = pd.DataFrame({'sinal': np.arange(10)})
    freq_amostragem = 10.0
    # freq_corte_hz equal to Nyquist frequency should raise ValueError
    with pytest.raises(ValueError):
        aplicar_filtro_butterworth(
            df,
            'sinal',
            freq_corte_hz=freq_amostragem / 2,
            freq_amostragem_hz=freq_amostragem,
        )

