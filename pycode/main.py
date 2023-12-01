import environ
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', help='return a lis of numbers')
parser.add_argument('--language', help='python')

args = parser.parse_args()


# load the .env file
environ.Env.read_env('../.env')

env = environ.Env(
    # set casting, default value
    OPEN_IA_KEY=(str, '')
)

# access the variable


llm = openai.OpenAI(
    openai_api_key=env('OPEN_IA_KEY'),
)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}.",
    input_variables=["language", "task"],
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} function:\n{code}",
    input_variables=["language", "code"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key='code')
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key='test')

chain = SequentialChain(
    chains=[
        code_chain,
        test_chain,
    ],
    input_variables=["task", "language"],
    output_variables=["code", "test"],
)

result = chain(
    {
        'language': args.language,
        'task': args.task,
    }
)

print(result['code'])
print(result['test'])
