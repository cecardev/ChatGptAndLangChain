import environ
from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

code_chain = LLMChain(llm=llm, prompt=code_prompt)

result = code_chain(
    {
        'language': 'python',
        'task': 'a list of numbers'
    }
)

print(result['text'])
