application = dict(
    ENVIRONMENT='DEV',
    APPLICATION_ROOT='/',
    BE_ROOT='http://localhost:8080/',
    THRESHOLD_OPTIMIZER=1,  # Possible values 0 or 1. Caution: Setting this property to one may reduce performance.
    EVALUATION_METHOD=dict(METHOD_NAME='KFold', METHOD_VALUE=10)  # Possible values "KFold=10" or "SplitByRatio=0.66"
)