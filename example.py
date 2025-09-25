from alio import AliO

alio_loss = AliO(
    num_samples=2,
    lag=1,
    time_loss='mse',
    freq_loss='mse'
)