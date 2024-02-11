from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def callbacks(outdir):
    early_stopping_monitor = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    checkpoint = ModelCheckpoint(
        outdir, save_best_only=True, monitor="val_loss", mode="min"
    )

    # return {"early_stopping_monitor": early_stopping_monitor, "checkpoint": checkpoint}
    return {"checkpoint": checkpoint}
