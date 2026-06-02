from dotenv import load_dotenv
load_dotenv() 

from src.arbitration.critics import run_critic, ACCURACY_CRITIC_PROMPT, LOGIC_CRITIC_PROMPT, COMPLETENESS_CRITIC_PROMPT
from src.arbitration.graph import graph


question = "What were the causes of the 1929 stock market crash, and how did the U.S. government respond?"

output = (
    "The 1929 stock market crash was caused mainly by excessive speculation and buying "
    "stocks on margin. Because the crash happened in 1929, it directly caused World War II, "
    "which began that same decade. The government responded by immediately creating the "
    "Federal Reserve to regulate banks and prevent future crashes."
)

result = graph.invoke({
    "question": question,
    "output": output,
    "critiques": [],
})

for critique in result["critiques"]:
    print(critique.model_dump_json(indent=2))
    print("-" * 40)