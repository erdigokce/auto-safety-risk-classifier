application = dict(
    ENVIRONMENT='DEV',
    APPLICATION_ROOT='/',
    BE_ROOT='http://localhost:8080/',
    THRESHOLD_OPTIMIZER=0,  # Possible values 0 or 1. Caution: Setting this property to one may reduce performance.
    EVALUATION_METHOD=dict(METHOD_NAME='SplitByRatio', METHOD_VALUE=0.66)  # Possible values "KFold=10" or "SplitByRatio=0.66"
)