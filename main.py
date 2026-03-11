from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    info="""
    Japan[a] is an island country in East Asia. Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered to the west by the Sea of Japan and extends from the Sea of Okhotsk in the north to the East China Sea in the south. The Japanese archipelago consists of four major islands alongside 14,121 smaller islands. Japan is divided into 47 administrative prefectures and eight traditional regions, and around 75% of its terrain is mountainous and heavily forested, concentrating its agriculture and highly urbanized population along its eastern coastal plains. With a population of almost 123 million as of 2026, it is the world's 11th most populous country. Tokyo is the country's capital and largest city.
    """

    model_prompt="""
    based on the information {info} provide 2 facts on this country 
    """
    country_prompt=PromptTemplate(template=model_prompt, input_variables=["info"])

    model=ChatOpenAI(model="gpt-4o-mini", temperature=0)
    country_chain=country_prompt | model
    result=country_chain.invoke({"info":info})
    print(result.content)
    
if __name__ == "__main__":
    main()
