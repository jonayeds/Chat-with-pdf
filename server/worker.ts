import { Worker } from "bullmq";
import { QdrantVectorStore } from "@langchain/qdrant";
import { Document } from "@langchain/core/documents";
import { PDFParse } from "pdf-parse";
import { CharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

const worker = new Worker(
  "file-upload-queue",
  async (job) => {
    const { filename, source, path } = JSON.parse(job.data);
    console.log(`Processing file ${filename} from ${source} at ${path}`);

    const parser = new PDFParse({ url: path });
    const doc = (await parser.getText()).text;
    const textSplitter = new CharacterTextSplitter({
      chunkSize: 300,
      chunkOverlap: 0,
    });
    const texts = await textSplitter.splitText(doc);
    console.log(texts.length);
    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACE_API_KEY,
      model: "BAAI/bge-base-en-v1.5",
    });

    const res = await embeddings.embedQuery(texts[0]!);
    console.log(res);
    const docs = texts.map((t) => new Document({ pageContent: t }));
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: "http://localhost:6333",
        collectionName: "pdf-collection",
      }
    );
    await vectorStore.addDocuments(docs);
    console.log("all docs are added to vector database");
  },
  {
    connection: {
      host: "localhost",
      port: 6379,
    },
    concurrency: 5,
  }
);
