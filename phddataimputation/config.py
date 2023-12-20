def config1():
    return [
        {"features": f, "neurons": n, "scaling": s}
        for f in [1, 2]
        for n in [128, 256]
        for s in [True, False]
    ]


def config2():
    return [
        {"features": f, "neurons": n, "scaling": s}
        for f in [3, 4]
        for n in [128, 256]
        for s in [True, False]
    ]


def config3():
    return [
        {"features": f, "neurons": n, "scaling": s}
        for f in [5, 6]
        for n in [128, 256]
        for s in [True, False]
    ]


def config4():
    return [
        {"features": f, "neurons": n, "scaling": s}
        for f in [7, 8]
        for n in [128, 256]
        for s in [True, False]
    ]
