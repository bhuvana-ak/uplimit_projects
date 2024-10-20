import os
import argparse
from pathlib import Path
from datasets import load_dataset as hf_load_dataset
import argilla as rg
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from distilabel.llms import TransformersLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts, TextGenerationToArgilla
from distilabel.steps.tasks import TextGeneration
from google.colab import userdata

class LLMEvaluator:
    def __init__(self):
        self.api_url = os.getenv('ARGILLA_API_URL')
        self.api_key = os.getenv('ARGILLA_API_KEY')
        self.workspace = os.getenv('ARGILLA_WORKSPACE', 'admin')
        self.dataset_name = os.getenv('ARGILLA_DATASET_NAME', 'DIBT_10k_prompts')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

        if not self.api_url or not self.api_key:
            raise ValueError("ARGILLA_API_URL and ARGILLA_API_KEY must be set in the environment variables.")

    def setup_argilla(self):
        rg.init(api_url=self.api_url, api_key=self.api_key)
        
        # Check if the dataset already exists
        try:
            existing_dataset = rg.FeedbackDataset.from_argilla(name=self.dataset_name, workspace=self.workspace)
            print(f"Dataset '{self.dataset_name}' already exists in workspace '{self.workspace}'")
        except rg.errors.NotFoundError:
            # Dataset doesn't exist, so we create it
            new_dataset = rg.FeedbackDataset(
                fields=[
                    rg.TextField(name="id"),
                    rg.TextField(name="instruction"),
                    rg.TextField(name="generation"),
                ],
                questions=[
                    rg.LabelQuestion(
                        name="quality",
                        labels=["👎", "👍"],
                        title="Quality of the generated text",
                    )
                ],
            )
            new_dataset.push_to_argilla(name=self.dataset_name, workspace=self.workspace)
            print(f"New dataset '{self.dataset_name}' created in workspace '{self.workspace}'")

    def download_model(self, model_name, local_model_path):
        print(f"Downloading {model_name} to {local_model_path}...")
        snapshot_download(repo_id=model_name, local_dir=local_model_path)
        print(f"Model {model_name} has been successfully downloaded and loaded.")
        total_size = sum(f.stat().st_size for f in Path(local_model_path).glob('**/*') if f.is_file())
        print(f"Total size of the downloaded model: {total_size / 1e9:.2f} GB")

    def create_pipeline(self, model_path=None, use_openai=False):
        filtered_dataset = hf_load_dataset("DIBT/10k_prompts_ranked", split="train").filter(
            lambda r: float(r["avg_rating"]) >= 4 and int(r["num_responses"]) >= 2
        )
        filtered_dataset_12 = filtered_dataset.select(range(12))

        with Pipeline(
            name="prefs-with-llm",
            description="Pipeline for building preference datasets using LLM",
        ) as pipeline:
            load_data = LoadDataFromDicts(
                name="load_dataset",
                data=filtered_dataset_12,
                output_mappings={"prompt": "instruction"},
            )
            
            if use_openai:
                llm = OpenAILLM(model="gpt-4")
            else:
                llm = TransformersLLM(
                    model=model_path,
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                    model_kwargs={"low_cpu_mem_usage": True},
                )
            
            text_generation = TextGeneration(name="text_generation", llm=llm)

            to_argilla = TextGenerationToArgilla(
                name="text_generation_to_argilla",
                dataset_name=self.dataset_name,
                dataset_workspace=self.workspace,
            )
            load_data >> text_generation >> to_argilla

        return pipeline

    def run_pipeline(self, pipeline, use_openai=False):
        parameters = {
            "load_dataset": {"batch_size": 16},
            "text_generation": {
                "llm": {
                    "generation_kwargs": {
                        "temperature": 0.7,
                    }
                }
            },
            "text_generation_to_argilla": {
                "api_url": self.api_url,
                "api_key": self.api_key,
                "dataset_name": self.dataset_name,
                "dataset_workspace": self.workspace,
            },
        }
        
        if not use_openai:
            parameters["text_generation"]["llm"]["generation_kwargs"].update({
                "max_new_tokens": 512,
                "do_sample": True,
                "top_p": 0.95,
                "top_k": 50,
            })
        
        distiset = pipeline.run(parameters=parameters)
        return distiset

def main(args):
    evaluator = LLMEvaluator()
    
    # Setup Argilla
    evaluator.setup_argilla()

    # Download model if not using OpenAI
    if not args.use_openai:
        evaluator.download_model(args.model_name, args.local_model_path)

    # Create and run pipeline
    pipeline = evaluator.create_pipeline(
        model_path=args.local_model_path if not args.use_openai else None,
        use_openai=args.use_openai
    )
    distiset = evaluator.run_pipeline(pipeline, args.use_openai)

    print("Pipeline execution completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Evaluation Script")
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name")
    parser.add_argument("--local_model_path", default="/content/tinyllama-1.1b-chat", help="Local model path")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI model instead of local model")
    
    args = parser.parse_args()
    
    main(args)

# Terminal commands:
# To install required packages:
# pip install -qqq huggingface_hub argilla accelerate
# pip install -qqq --upgrade "distilabel[huggingface]"
# pip install openai

# To run the script:
# python script_name.py [--use_openai]

# To run lm-evaluation-harness:
# git clone https://github.com/EleutherAI/lm-evaluation-harness
# cd lm-evaluation-harness
# pip install -e .
# lm_eval --model hf --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" --tasks hellaswag --device cuda --batch_size auto:4 --output_path hellaswag_test
