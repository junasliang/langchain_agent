from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-agent!")
    information = """
    伊隆·里夫·馬斯克（英語：Elon Reeve Musk /ˈiːlɔːn/；1971年6月28日—），漢名馬誼郎[註 3]，世界首富、商業大亨、英國皇家學會會士[註 4]、美國工程院院士[6]。他是SpaceX的創始人、董事長、執行長、首席工程師，特斯拉投資人、執行長、產品設計師、前董事長，無聊公司創始人，Neuralink、OpenAI聯合創始人，同時也是X公司的技術長、董事長。2022年馬斯克以2190億美元財富成為世界首富。[7]2025年10月1日15時30分，馬斯克財富突破5000億美元，成為全球首位半萬億美元富豪，標誌其商業帝國再創新高。2026年2月其投資組合整併，身家已經突破8000億美元[8]。

    馬斯克在南非川斯瓦省普利托利亞長大，曾短暫就讀於普利托利亞大學，後於18歲移居加拿大就讀女王大學。兩年後，他轉學到賓夕法尼亞大學，並獲得經濟學、物理學的學士學位。1995年，他搬到加利福尼亞州並就讀於史丹佛大學，而後決定從商。他與弟弟金巴爾共同創辦了網路軟體公司Zip2。1999年，康柏公司以3.07億美元收購了這家初創企業。同年，馬斯克聯合創辦了線上銀行X.com。2000年，該公司與Confinity合併為PayPal。2002年，EBay以15億美元買下PayPal。

    2002年，馬斯克創辦SpaceX，並擔任董事長、執行長、技術長，該公司主要負責太空運輸、航太製造。2004年，他加入電動車製造商特斯拉，並擔任董事長與產品設計師，2008年兼任執行長。2006年，他協助創立太陽能服務公司SolarCity，該公司後成為特斯拉子公司特斯拉能源。2015年，他聯合創辦了非營利公司OpenAI，用於研究和推動友善人工智慧。2016年，他聯合創辦了神經科技公司Neuralink，該公司專注於開發人機介面。同年，他成立了無聊隧道施工公司，用於研發超迴路列車。[註 5]2022年10月27日，馬斯克以440億美元收購社群平台Twitter，日後改組為X。2021年10月，美國商業雜誌《富比士》宣布馬斯克財富達到2700億美元，成為該雜誌統計史上最富有的人[9]。2024年11月，美國總統當選人唐納·川普宣布委任馬斯克為總統高級顧問，領導新創立的政府效率部[10]。

    馬斯克被認為曾發表一些誤導性或違背科學的言論以及傳播關於COVID-19的錯誤資訊而受到批評。此外，馬斯克在人工智慧、加密貨幣、大眾運輸等方面的觀點亦受到部分專家的批評。    
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
