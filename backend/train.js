import DiseasePrediction from "./helper/DiseasePrediction.js";
import fs from "fs";

let disease =
  `hypertensive  disease,diabetes,depression  mental,depressive disorder,coronary  arteriosclerosis,coronary heart disease,pneumonia,failure  heart congestive,accident  cerebrovascular,asthma,myocardial  infarction,hypercholesterolemia,infection,infection  urinary tract,anemia,chronic  obstructive airway disease,dementia,insufficiency  renal,confusion,degenerative  polyarthritis,hypothyroidism,anxiety  state,malignant  neoplasms,primary malignant neoplasm,acquired  immuno-deficiency  syndrome,HIV,hiv infections,cellulitis,gastroesophageal  reflux disease,septicemia,systemic  infection,sepsis (invertebrate),deep  vein thrombosis,dehydration,neoplasm,embolism  pulmonary,epilepsy,cardiomyopathy,chronic  kidney failure,carcinoma,hepatitis  C,peripheral  vascular disease,psychotic  disorder,hyperlipidemia,bipolar  disorder,obesity,ischemia,cirrhosis,exanthema,benign  prostatic hypertrophy,kidney  failure acute,mitral  valve insufficiency,arthritis,bronchitis,hemiparesis,osteoporosis,transient  ischemic attack,adenocarcinoma,paranoia,pancreatitis,incontinence,paroxysmal  dyspnea,hernia,malignant  neoplasm of prostate,carcinoma prostate,edema  pulmonary,lymphatic  diseases,stenosis  aortic valve,malignant  neoplasm of breast,carcinoma breast,schizophrenia,diverticulitis,overload  fluid,ulcer  peptic,osteomyelitis,gastritis,bacteremia,failure  kidney,sickle  cell anemia,failure  heart,upper  respiratory infection,hepatitis,hypertension  pulmonary,deglutition  disorder,gout,thrombocytopaenia,hypoglycemia,pneumonia  aspiration,colitis,diverticulosis,suicide  attempt,Pneumocystis  carinii pneumonia,hepatitis  B,parkinson  disease,lymphoma,hyperglycemia,encephalopathy,tricuspid  valve insufficiency,Alzheimer's  disease,candidiasis,oral  candidiasis,neuropathy,kidney  disease,fibroid  tumor,glaucoma,neoplasm  metastasis,malignant  tumor of colon,carcinoma colon,ketoacidosis  diabetic,tonic-clonic  epilepsy,tonic-clonic seizures,respiratory  failure,melanoma,gastroenteritis,malignant  neoplasm of lung,carcinoma of lung,manic  disorder,personality  disorder,primary  carcinoma of the liver cells,emphysema  pulmonary,hemorrhoids,spasm  bronchial,aphasia,obesity  morbid,pyelonephritis,endocarditis,effusion  pericardial,pericardial effusion body substance,chronic  alcoholic intoxication,pneumothorax,delirium,neutropenia,hyperbilirubinemia,influenza,dependence,thrombus,cholecystitis,hernia  hiatal,migraine  disorders,pancytopenia,cholelithiasis,biliary  calculus,tachycardia  sinus,ileus,adhesion,delusion,affect  labile,decubitus  ulcer`.split(
    ","
  );

const getTrainingData = async () => {
  let json = JSON.parse(fs.readFileSync("output.json", "UTF-8"));
  let array = json.array;
  let xInputs = [];
  let yInputs = [];

  for (let i of array) {
    let x = i.inputs.split(",").map(Number);
    let y = i.outputs.split(",").map(Number);
    xInputs.push(x);
    yInputs.push(y);
  }

  const predictor = new DiseasePrediction();
  predictor.compile();
  predictor.readJsonWeights();
  let counter = 0;
  for (let curr in disease) {
    await predictor.train(xInputs, yInputs);
    let result = predictor.predict([xInputs[curr]]);
    let i = 0;
    let prevVal = -1;
    result.map((val, idx) => {
      if (val > prevVal) {
        i = idx;
        prevVal = val;
      }
    });
    if (disease[i] == disease[curr]) {
      counter++;
    }
  }

  console.log(counter / disease.length, counter);
};

getTrainingData();
