from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

if __name__ == "__main__":
    load_dotenv()

    llm = ChatHuggingFace(temperature=0, llm=HuggingFacePipeline.from_model_id(
        model_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        ),
    ))
