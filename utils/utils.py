import inspect

def get_valid_kwargs(cls, kwargs):
    # Obtém a assinatura do método __init__ da classe
    sig = inspect.signature(cls.__init__)
    # Pega os nomes dos parâmetros (removendo o 'self')
    valid_keys = set(sig.parameters.keys()) - {'self'}
    # Retorna um dicionário filtrado apenas com os parâmetros válidos
    return {k: v for k, v in kwargs.items() if k in valid_keys}