application = dict(
    ENVIRONMENT='DEV',
    THRESHOLD_OPTIMIZER=0,  # Possible values 0 or 1. Caution: Setting this property to "1" may reduce performance.
    EVALUATION_METHOD=dict(METHOD_NAME='SplitByRatio', METHOD_VALUE=0.66)  # Possible values "KFold=10" or "SplitByRatio=0.66"
)
