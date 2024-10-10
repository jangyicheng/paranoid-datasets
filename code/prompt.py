from collections import defaultdict



#所有prompt对象的基类
class Prompt:
    def __init__(self, base_prompt):
        self.base_prompt = base_prompt
    def prompt(self, category=None):#按类别输出特定prompt
        if category!=None:
            return self._select_prompt_by_category(category)

        return self.base_prompt#输出原始prompt

    def _select_prompt_by_category(self, category):
        raise NotImplementedError("Subclasses should implement this method.")


class QuestionPrompt(Prompt):
    def __init__(self):
        base_prompt = """
# Role Setting
You are a question-design expert specializing in creating questions in dialogue based on different images,
materials and conversation history to simulate everyday conversations. Your task is to design multiple relatively simple questions based on given images, their corresponding captions provided by the user, and conversation history.
Questions can be about images as well as text descriptions, or they can be further in-depth questions about the last user's response.

Key Requirements
- The questions must not reveal the specific content of the images and materials; instead, use "<image-1>" and "<image-2>" to refer to them. For images without caption annotations, "<caption>" will be replaced with "<none>". In this case, you need to interpret the image on your own.
- Images are entered in order, use <image-i> to refer to the ith image, e.g., when referencing the 3-rd image in a sequential entry, use <image-3> to refer to it.
- Based on the image content, you can design questions from the following perspectives: {task_types}.
#注意这里的{task_types}会被重写

- Do not mention the use of supplementary materials or text discriptions in the questions！
- The questions should test the ability to understand multiple images simultaneously, covering various aspects of multi-image comprehension. Each question must cover two or more image contents simultaneously,the question should involve the relationships between the contents of multiple images.
- Ensure that each question can relate to multiple images as much as possible.
- Add a "Q:[QUESTION TYPE]" before each question, fields in '[QUESTION TYPE]' depend on the specific type of issue
- You only need to provide the question in the format mentioned below, without answering any other content.

## Below are the input formats for the photo captions and supplementary materials you will need to refer to:
Input:
images: <image-1> <image-2> <image-3> <image-4>
texts:
<image-1>: <discription1>
<image-2>: <discription2>
<image-3>: <none>
<image-4>: <none>
##Example Prompt:
Output:
"[PRICE COMPARISON] Based on the prices shown, which book in <image-1> and <image-4> is the most expensive, and which is the least expensive? How do their prices compare to the books in <image-2> and <image-3>?",
"[LAYOUT ANALYSIS] How do the layout and design of the book listings in <image-1> and <image-3> differ from each other? What might be the advantages of each layout?",
"[IN-DEPTH QUESTIONS] In your analysis of <image-1> and <image-3>, you mentioned that the layouts differ, with <image-1> having a sidebar with detailed filter options and <image-3> focusing more on book listings with a grid layout. Can you explain how each layout might affect a user's browsing and decision-making process in an online bookstore?",
"[RATINGS AND REVIEWS] Considering the star ratings given to the books in <image-2> and <image-3>, how do customers' preferences appear to differ between these books and those in <image-4>?" """
        super().__init__(base_prompt)

    def _select_prompt_by_category(self, category="default"):
        task_templates = {
            #默认任务类型
            "default": """description of the image content, comparison of the image content, social, cultural, 
            or historical significance of the image, emotional expression in the image, symbolic meaning of elements in the image, 
            interpretation of actions or activities in the image, textual and graphical content in the image, sequencing of consecutive actions in the image, 
            description of the location of a specific object in the image, 
            and relationships between objects or people in the image""",

            #文档类任务
            "DOCVQA":"""
            [CONTENT COMPARISON]: Compare the information presented in different images.
            [SEQUENTIAL UNDERSTANDING]: Evaluate the ability to follow a sequence of events or processes across multiple images.
            [CROSS-REFERENCING]: Test the ability to cross-reference information between different pages.
            [CONTEXTUAL INFERENCE]: Ask about inferred information based on the context provided by multiple images.
            [SUMMARIZATION]: Request a summary that encapsulates the content from multiple images.
            [DETAIL EXTRACTION]: Focus on extracting specific details from different images and explaining their importance.
            [ANALYTICAL REASONING]: Pose analytical questions that require synthesizing information from various images.
            [HISTORICAL CONTEXT]: Ask how historical context or previous information relates to current content.
            """,

            # 图表类任务
            "CHART_QA":"""
            [ACTIVITY INFERENCE] : Identifying actions or activities occurring across multiple images.
            [COMPARATIVE ANALYSIS] : Comparing and contrasting elements across multiple images or tables.
            [TREND IDENTIFICATION] : Recognizing trends or patterns across multiple tables or images.
            [CROSS-REFERENCING]: Test the ability to cross-reference information between different charts.
            [DATA CORRELATION] : Finding correlations or relationships between data points in multiple tables.
            [CONTENT COMPREHENSION] : Understanding and summarizing the content presented in multiple images or tables.
            [HYPOTHESIS GENERATION] : Formulating hypotheses based on the information in multiple images or tables.
            [ANOMALY DETECTION] : Identifying anomalies or outliers in multiple images or tables.
            [SYNTHESIS AND INTEGRATION] : Integrating information from multiple sources to derive conclusions.
            [PREDICTION AND FORECASTING] : Making predictions based on the data trends in multiple tables.
            """,

            # 连续动作任务
            "NEXTQA":"""
            [SEQUENTIAL DESCRIPTION]: Describing the sequence of actions occurring across multiple images.
            [CAUSE AND EFFECT]: Explaining the reasons or outcomes of actions depicted in multiple images.
            [PREDICTION AND INFERENCE]: Predicting the next image based on the sequence of previous images.
            [COMPARATIVE ANALYSIS]: Comparing and contrasting elements across multiple images.
            [STORY SYNTHESIS]: Summarizing a story or event based on the content of multiple images.
            [DETAIL EXTRACTION]: Extracting specific details from individual images within a series.
            [IDENTIFICATION OF ELEMENTS]: Identifying the main characters or objects and their actions in multiple images.
            [LOGICAL REASONING]: Inferring the impact of changes in one image on subsequent images.
            [ACTION RECOGNITION]: Identifying the specific actions of people or objects in an image.
            [OBJECT TRACKING]: Tracking the movement trajectory of specific objects or people across a sequence of images.
            """

        }


        default_type="""description of the image content, comparison of the image content, social, cultural, 
            or historical significance of the image, emotional expression in the image, symbolic meaning of elements in the image, 
            interpretation of actions or activities in the image, textual and graphical content in the image, sequencing of consecutive actions in the image, 
            description of the location of a specific object in the image, 
            and relationships between objects or people in the image."""
        task_type_description = task_templates.get(category, default_type)
        appendix = defaultdict(
            lambda: "", {
                "DOCVQA": "The set of images provided to you is derived from a set of closely related documents.Specific information about the image is given in the text description.",
                "CHART_QA": "The set of images provided to you comes from a set of charts with similar content",
                "PubMed": "The set of images provided to you is derived from a set of medical images with similar content.",
                "geo": "The set of images provided to you is derived from a set of geometric images with similar content",
                "COMICS": "The set of images provided to you is an excerpt from a set of consecutive segments from the comic strip",
                "LECTUREBANK": "The set of images provided to you is a continuous set of images excerpted from online lecture files covering 5 different areas including natural language processing (nlp), machine learning (ml), artificial intelligence (ai), deep learning (dl) and information retrieval (ir).",
                "NEXTQA": "The set of images provided to you comes from the same set of consecutive videos.",
                "MIND2WEB": "The set of images provided to you comes from a split of a long screenshot of the same web page.",
                "Wikiart": """The images provided to you all come from the same author of the same genre of painting, or they are from anonymous authors""",
                "MAGICBRUSH": """The images provided to you are paired with subtle differences, which are indicated in the text descriptions""",
                "OCR": """The images provided to you are selected from the covers of two books.""",
                "FOOD": """The images provided to you are all of the same category of food. """,
                "VASR": """ The images provided to you all depict movements or behavioural pattern of the same type.""",
                "COCO": """ Each of the images in this set contains objects belonging to the same category.""",
                "OFFICEHOME": """ Images 1-4 show the same object from 4 different domains: Artistic images, Clip Art, Product images and Real-World images.""",
                "Alfred": """ This set of images shows a set of actions in three-dimensional space, and the text corresponding to each image describes the action that is about to take place in the next scene.""",

            })
        #几何类提问prompt需要进行重写
        geo_prompt = """ ##Role Setting
        You are a question-design expert specializing in creating math questions that relate multiple images simultaneously in dialogue based on different images,
        materials and conversation history . Your task is to design multiple relatively simple questions based on given images, their corresponding captions provided by the user, and conversation history.
        ##Key Requirements
        -The questions must not reveal the specific content of the images and materials; instead, use "<image-1>" and "<image-2>" to refer to them. 
        -For a single image, the provided auxiliary information will appear in the form of question-answer pairs.
        -Don't ask questions that have already appeared in the conversation history!
        -To simulate everyday conversations, your questions must be built upon the foundation of the previous question-and-answer conversation history, frequently referencing past dialogues, and even posing new questions regarding previous conversation history.
        -Do not mention the use of supplementary materials or image captions in the questions.
        -Add a "Q:[QUESTION TYPE]" before each question, fields in '[QUESTION TYPE]' depend on the specific type of issue
        -The designed problem must relate both pictures, examining the understanding of the relationship between them, such as the similarities and differences between their solutions, examining the relationship between the knowledge points, etc.
        -Based on the image content, you can design questions from the following perspectives: Geometric relationships, geometric solutions, geometric understanding.
        -You only need to provide the question in the format mentioned below, without answering any other content.
        -Below are the input formats for the photo captions and supplementary materials you will need to refer to:
        Input:
        images: <image-1> <image-2> <image-3> <image-4>
        texts:
        <image-1>: <discription1>
        <image-2>: <discription2>
        <image-3>: <none>
        <image-4>: <none>"""
        if category == 'geo':
            return geo_prompt

        return self.base_prompt.replace("{task_types}", task_type_description)+appendix[category]



class AnswerPrompt(Prompt):
    def __init__(self):
        super().__init__("""# Role Setting
You are an answer expert, specializing in answering questions based on images and text descriptions. Your task is to answer
multiple questions based on the images provided by the user and conversation history.
## Key Requirements
- You need to provide detailed,high-quality and accurate answers to each question based on the given images and text descriptions.
- Add a "A:" before each answer
- Your answers should be as accurate as possible, with each answer being around 150 words long, less than 200 words long.
- Images are entered in order, use <image-i> to refer to the ith image, e.g., when referencing the first image in a sequential entry, use <image-1> to refer to it.
- Your answer should demonstrate a high level of Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy, Image Relationship Understanding.
- Your answer should be highly logical, rigorous and descriptive.
- Do not repeat the question; just provide the answer directly.
## Example Prompt:
- Input:
Images: <image-1> <image-2>
Text:
<image-1>: <none>
<image-2>: <none>
Q: [ATTRIBUTE SIMILARITY]What are the common features of the sculptures in <image-1> and <image-2>?
- Output:
A: The common features of these two sculptures are:
Similar material: Both are made of a similar silver metallic material, with a smooth and reflective surface.
Similar artistic style: Both sculptures employ a highly realistic style, depicting the form and details of the figures meticulously.
""")

    def _select_prompt_by_category(self, category):
        prompts = {
            "DOCVQA": "The set of images provided to you is derived from a set of closely related documents.No need to be too precise in answering questions about numbers, entity identification, etc.Specific information about the image is given in the text description.",
            "CHART_QA": "The set of images provided to you comes from a set of charts with similar content",
            "PubMed": "The set of images provided to you is derived from a set of medical images with similar content.",
            "geo": """The set of images provided to you is derived from a set of geometric images with similar content """,
            "COMICS": """The set of images provided to you is an excerpt from a set of consecutive segments from the comic strip""",
            "LECTUREBANK": """ The set of images provided to you is a continuous set of images excerpted from online lecture files covering 5 different areas including natural language processing (nlp), machine learning (ml), artificial intelligence (ai), deep learning (dl) and information retrieval (ir).""",
            "NEXTQA": """ The set of images provided to you comes from the same set of consecutive videos.""",
            "MIND2WEB": """ The set of images provided to you comes from a split of a long screenshot of the same web page.""",
            "Wikiart":"""The images provided to you all come from the same author of the same genre of painting, or they are from anonymous authors""",
            "MAGICBRUSH":"""The images provided to you are paired with subtle differences, which are indicated in the text descriptions""",
            "OCR":"""The images provided to you are selected from the covers of two books.""",
            "FOOD":"""The images provided to you are all of the same category of food. """,
            "VASR":""" The images provided to you all depict movements or behavioural pattern of the same type .""",
            "COCO":""" Each of the images in this set contains objects belonging to the same category.""",
            "OFFICEHOME": """ Images 1-4 show the same object from 4 different domains: Artistic images, Clip Art, Product images and Real-World images.""",
            "Alfred": """ This set of images shows a set of actions in three-dimensional space, and the text corresponding to each image describes the action that is about to take place in the next scene.""",

        }


        return self.base_prompt+prompts.get(category, " ")

class EvaluatePrompt(Prompt):
    def __init__(self):
        super().__init__(
            """  You are an expert in evaluating the quality of responses to questions. You assess and rate the quality of responses to given images and related questions for each set.
                    **Instructions:**
    
                    1. Evaluation should be based on the following aspects: Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy, and Image Relationship Understanding.
                    2. Ensure to strictly follow the output format below without any extra responses.
                    3. All questions involve multiple images simultaneously, assessing the ability to understand the context of multiple images. Your evaluation should consider this ability.
                    4. The evaluation score ranges from 1 to 10, with 1 indicating almost no understanding, 8 indicating an understanding at the level of an average human, and 10 indicating an understanding at the level of an expert.
    
                    ### Prompt Format:
    
                    **Input:**
    
                    Images: <image-1> <image-2> <image-3> <image-4> ...
                    Text:
                    <image-1>: <caption1>
                    <image-2>: <caption2>
                    <image-3>: <caption3>
                    <image-4>: <caption4>
                    ...
    
                    Question 1: <question1>
                    Question 2: <question2>
                    ...
                    Question n: <questionn>
    
                    Answer 1: <answer1>
                    Answer 2: <answer2>
                    ...
                    Answer n: <answern>
    
                    **Output:**
    
                    ### Creativity
                    <Evaluate the creativity, including whether the response shows a unique and imaginative understanding of the images. Example: The response shows a creative interpretation of image details but could elaborate more on how these elements convey a specific theme. **Score: 7**>
    
                    ### Richness
                    <Evaluate the richness, whether it includes specific details and provides deep insights. Example: The response includes specific details, such as the large skeleton holding the woman by her hair... However, it could provide deeper insights into how these elements effectively contribute to the horror theme. **Score: 7**>
    
                    ### Visual Perception
                    <Evaluate the visual perception, whether key elements in the images are accurately identified. Example: The response correctly identifies key elements in both images, such as the skeleton, the woman, the gothic house, the large red hands, and the woman's expression... Although the response includes relevant information, it could further enrich the description by providing more intricate details, such as the background or additional characters in the images. **Score: 7**>
    
                    ### Logical Coherence
                    <Evaluate the logical coherence, whether the structure is reasonable and the response aligns with the question. Example: The response is logically structured and closely related to the question, demonstrating a deep understanding of the images. **Score: 8**>
    
                    ### Answer Accuracy
                    <Evaluate the accuracy of the answer, whether it correctly answers the question and is consistent with the image and text information. Example: The response accurately answers all the questions and is completely consistent with the image and text information. **Score: 8**>
    
                    ### Image Relationship Understanding
                    <Evaluate the understanding of image relationships, whether the content of multiple images is accurately distinguished and related. Example: The response accurately distinguishes between the two images, identifying unique features of each while also pointing out their commonalities. By discussing typical horror motifs and the overall mood created by each image, it effectively relates the images to the horror theme. **Score: 8**>
    
                    ### Overall Score
                    <Consider the scores in each dimension to evaluate the overall quality of the response. Example: Considering the scores in each dimension, the response is accurate, logically structured, and rich in specific details but could use more creativity and depth to perfectly capture the horror theme. **Overall Score: 7**>
    
                    {'Creativity': 6, 'Richness': 7, 'Visual Perception': 7, 'Logical Coherence': 7, 'Answer Accuracy': 8, 'Image Relationship Understanding': 8, 'Overall Score': 7}
    """)





