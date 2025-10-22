import express from "express";
import cors from "cors";
import multer from "multer";
import { Queue } from "bullmq";
import { QdrantVectorStore } from "@langchain/qdrant";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { ChatGroq } from "@langchain/groq";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "openai/gpt-oss-20b",
});

const queue = new Queue("file-upload-queue", {
  connection: {
    host: "localhost",
    port: 6379,
  },
});

const app = express();

app.use(cors());

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({ storage: storage });

app.post("/upload/pdf", upload.single("pdf"), async (req, res) => {
  await queue.add(
    "file-ready",
    JSON.stringify({
      filename: req.file?.filename,
      source: req.file?.destination,
      path: req.file?.path,
    })
  );
  res.json({ message: "File uploaded successfully" });
});

app.get("/chat", async (req, res) => {
  const userQuery = req.query.message as string;
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: "BAAI/bge-base-en-v1.5",
  });
  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddings,
    {
      url: "http://localhost:6333",
      collectionName: "pdf-collection",
    }
  );
  const ret = vectorStore.asRetriever({
    k: 2,
  });
  const result = await ret.invoke(userQuery);
  const SYSTEM_PROMPT = `
    You are a helpful assistant that canswers questions about the user query based on the available context from pdf files
    Context\n${JSON.stringify(result).replace(/{/g, "[").replace(/}/g, "]")}
    `;
  const prompt = ChatPromptTemplate.fromMessages([
    {
      role: "system",
      content: SYSTEM_PROMPT,
    },
    {
      role: "user",
      content: "{user_query}",
    },
  ]);
  const chatResult = await prompt.invoke({
    user_query: userQuery,
  });
  const response = await llm.invoke(chatResult);
  return res.json({ answer: response, docs: result });
});

app.listen(8000, () => {
  console.log("Server is running on PORT 8000");
});
