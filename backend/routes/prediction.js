import express from "express";
import { predictDisease } from "../controller/prediction.js";

const router = express.Router();

router.post("/predictDisease", predictDisease);

export default router;
