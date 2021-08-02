import express from "express";
import cors from "cors";
import predict from "./routes/prediction.js";
let app = express();

app.use(express.json());
app.use(cors());
app.use(predict);

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => console.log(`Server Listening in port:${PORT}`));
