import express from "express";
import { predictDisease } from "../controller/prediction";

const router = express.Router();

router.post("/predictDisease", predictDisease);

export default router;
