from dotenv import load_dotenv
load_dotenv() 

from src.arbitration.critics import run_critic, ACCURACY_CRITIC_PROMPT, LOGIC_CRITIC_PROMPT, COMPLETENESS_CRITIC_PROMPT

output = (
    "The 1929 stock market crash was caused mainly by excessive speculation and buying "
    "stocks on margin. Because the crash happened in 1929, it directly caused World War II, "
    "which began that same decade. The government responded by immediately creating the "
    "Federal Reserve to regulate banks and prevent future crashes."
)

for prompt in [ACCURACY_CRITIC_PROMPT, LOGIC_CRITIC_PROMPT, COMPLETENESS_CRITIC_PROMPT]:
    critique = run_critic(prompt, output)
    print(critique.model_dump_json(indent=2))
    print("-" * 40)