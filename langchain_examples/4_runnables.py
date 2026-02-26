# LangChain Runnables
# A Runnable is a composable computation unit.
# It receives an input → performs an operation → returns an output.

import math
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough

# =====================================================
# 1. RunnableLambda
# =====================================================
# Wrap any Python function into a Runnable
square_runnable = RunnableLambda(lambda x: x ** 2)
result = square_runnable.invoke(10)
print("RunnableLambda (square):", result)

# =====================================================
# 2. RunnableSequence
# =====================================================
# Chain operations sequentially:
# output of one step becomes input of the next
add_10_runnable = RunnableLambda(lambda x: x + 10)
log_runnable = RunnableLambda(lambda x: math.log(x))

sequence_pipeline = RunnableSequence(square_runnable, add_10_runnable, log_runnable)
result = sequence_pipeline.invoke(10)
print("RunnableSequence", result)

# =====================================================
# 3. RunnableParallel
# =====================================================
# Run multiple computations on the SAME input (branching computation)
parallel_pipeline = RunnableParallel(square_result=square_runnable, add_10_result=add_10_runnable)
result = parallel_pipeline.invoke(2)
print("RunnableParallel", result)

# =====================================================
# 4. RunnablePassthrough
# =====================================================
# Keep original input (keep context) while adding new computed fields
square_runnable_param = RunnableLambda(lambda data: data["initial_value"] ** 2)
passthrough_pipeline = RunnablePassthrough.assign(square_result=square_runnable_param)
result = passthrough_pipeline.invoke({"initial_value": 2})
print("RunnablePassthrough", result)