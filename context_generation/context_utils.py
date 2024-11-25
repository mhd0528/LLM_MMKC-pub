import os
import torch
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False

# setting 1
# Function to get probabilities
def get_yes_no_probabilities(processor, generated_outputs):
    # Extract logits for each generation step
    token_scores = generated_outputs.scores  # List of logits for each step

    # Ensure that there's at least one step
    if len(token_scores) == 0:
        print("No token scores available.")
        return None, None, generated_outputs

    # Extract logits for the first token
    first_step_logits = token_scores[0]  # Tensor of shape (batch_size, vocab_size)

    # Apply softmax to get probabilities
    first_step_probs = torch.softmax(first_step_logits, dim=-1)

    # Extract token IDs for "Yes", "yes", "No", "no"
    yes_token_ids = [
        processor.tokenizer.encode("Yes", add_special_tokens=False)[0],
        processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    ]
    no_token_ids = [
        processor.tokenizer.encode("No", add_special_tokens=False)[0],
        processor.tokenizer.encode("no", add_special_tokens=False)[0]
    ]

    # Calculate combined probabilities for "Yes"/"yes" and "No"/"no"
    yes_probability = sum(first_step_probs[0, token_id].item() for token_id in yes_token_ids)
    no_probability = sum(first_step_probs[0, token_id].item() for token_id in no_token_ids)

    return yes_probability, no_probability


def get_prob_with_sentence(processor, model, prompt, image, device, model_mod="single"):
    # generate confidence of the image related to the entity
    if model_mod == "single":
        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,  # generate with sample may return conflict with prob
                # num_beams=5,
                max_length=256,
                min_length=1,
                # top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                # temperature=1,  # generate with greedy or not
                output_scores=True,
                return_dict_in_generate=True
            )
    elif model_mod == "multi":
        inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        # inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
        inputs['img_mask'] = torch.tensor([[1]])
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                img_mask=inputs['img_mask'],
                do_sample=False,
                max_length=50,
                min_length=1,
                set_min_padding_size=False,
                output_scores=True,
                return_dict_in_generate=True,
                sp_token=processor.tokenizer.img_place_token_id
            )

    # extract text and yes/no probability
    generated_text = processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )[0].strip()

    # use yes/no probability as confidence
    yes_prob, no_prob = get_yes_no_probabilities(processor, outputs)
    # yes_prob, no_prob = get_yes_no_cumulate_probabilities(outputs, processor, max_tokens_to_consider=1)

    return yes_prob, no_prob, generated_text


# Function to search for related entities and generate context for each image
def related_ent_img_search(ent_path, model, processor, prompt_list, device, model_mod="single"):
    context_list = []
    prompt_1, prompt_2, context_prompt = prompt_list
    # print(prompt_1, prompt_2, context_prompt)
    for image_path in os.listdir(ent_path):
        yes_probs, no_probs = [], []
        image_full_path = os.path.join(ent_path, image_path)
        # print(f"Content of '{image_full_path}'")
        try:
            image = Image.open(image_full_path).convert("RGB")
        except Exception as e:
            print(f"Error during opening image {image_full_path}:", e)
            continue
        # resize image if its size is 1, 1
        if image.size == (1, 1) or image.size[0] == 1 or image.size[1] == 1:
            print("Resizing the image, original size is 1*n or n*1, will cause encoder issue...")
            image = image.resize((10, 10))

        # generation and compute prob for subj
        yes_prob, no_prob, generated_text = get_prob_with_sentence(processor, model, prompt_1, image, device, model_mod)
        yes_probs.append(yes_prob)
        no_probs.append(no_prob)
        # log generated info
        # print(f"\t>>>> Identify ent1: {generated_text}")

        # generation and compute prob for sub
        yes_prob, no_prob, generated_text = get_prob_with_sentence(processor, model, prompt_2, image, device, model_mod)
        yes_probs.append(yes_prob)
        no_probs.append(no_prob)
        # print(f"\t>>>> Identify ent2: {generated_text}")

        # print(f"\t>>>> Image: {image_path} | Yes: {yes_prob:.6f}, No: {no_prob:.6f}")

        # generate additional context if both entity exists
        if yes_probs[0] > no_probs[0] and yes_probs[1] > no_probs[1]:
            if model_mod == "single":
                context_inputs = processor(
                    images=image, text=context_prompt, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    context_outputs = model.generate(
                        **context_inputs,
                        do_sample=True,  # generate with sample may return conflict with prob
                        # num_beams=5,
                        max_length=1024,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,  # generate with greedy or not
                        output_scores=True,
                        return_dict_in_generate=True
                    )
            elif model_mod == "multi":
                context_inputs = processor(images=[image], text=context_prompt, return_tensors="pt").to(device)

                context_inputs['pixel_values'] = context_inputs['pixel_values'].to(torch.bfloat16)
                context_inputs['img_mask'] = torch.tensor([[1]])
                context_inputs['pixel_values'] = context_inputs['pixel_values'].unsqueeze(0)
                context_inputs = context_inputs.to(device)

                with torch.no_grad():
                    context_outputs = model.generate(
                        pixel_values=context_inputs['pixel_values'],
                        input_ids=context_inputs['input_ids'],
                        attention_mask=context_inputs['attention_mask'],
                        img_mask=context_inputs['img_mask'],
                        do_sample=False,
                        max_length=50,
                        min_length=1,
                        set_min_padding_size=False,
                        output_scores=True,
                        return_dict_in_generate=True,
                        sp_token=processor.tokenizer.img_place_token_id
                    )

            # extract text and yes/no probability
            context_text = processor.batch_decode(
                context_outputs.sequences, skip_special_tokens=True
            )[0].strip()
            context_list.append(context_text)

            print(f"\t>>>> Image: {image_path} | Yes: {[round(yes_prob, 6) for yes_prob in yes_probs]}, No: {[round(no_prob, 6) for no_prob in no_probs]}")
            print(f"\t>>>> Generated context: {context_text}\n")
    return context_list

# setting 2
# function for multi-image summary
def get_image_relation_prompt(all_useful_images, subj_s, obj_s, replace_token="|"):
    '''
    Example (of what we want):

    query: Yambao (subj) | genre |
    context:
    instance of film | film
    country of origin | Mexico
    reverse of directed | Alfredo B Crevenna
    …
    We want:
    takes place in | Cuba (obj)
    '''

    if len(all_useful_images) == 0:
        return None

    token_replace_str = " ".join([f"image {idx}: <image{idx}>{replace_token}" + ("," if idx < len(all_useful_images) - 1 else "") for idx in range(len(all_useful_images))])
    prompt = (
        f"Analyze the set of images {token_replace_str}"
        "The model should primarily rely on the visual information in the images to understand the context."
        "In one complete, longer sentence, what is the relationship represented in the set of images?"
        "Don't provide trivial information. Only return useful information from the set of images."
        f"if entity {subj_s} and entity {obj_s} are not presented in the set of images, state that no relationship was found."
    )
    return prompt

# Get useful images
def get_useful_images(ent_path, model, processor, prompt_list, device="cuda", model_mod="multi", select_best_one=False):
    prompt_1, prompt_2, context_prompt = prompt_list

    useful_images = []
    best_image = None
    best_yes_prob = float('-inf')  # Initialize to negative infinity for comparison
    best_no_prob = None

    for image_path in os.listdir(ent_path):
        # Skip if the file is not an image
        if os.path.splitext(image_path)[-1] not in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
            continue
        yes_probs, no_probs = [], []
        image_full_path = os.path.join(ent_path, image_path)

        try:
            image = Image.open(image_full_path).convert("RGB")
        except PIL.UnidentifiedImageError as e:
            print(f"Error during opening image {image_full_path}:", e)
            continue

        # Resize image if its size is 1 * x or x * 1
        if image.size == (1, 1) or image.size[0] == 1 or image.size[1] == 1:
            # print("\t>>>>Resizing the image, original size is 1*1, will cause encoder issue...")
            image = image.resize((10, 10))

        # Generation and compute prob for subj
        yes_prob_1, no_prob_1, generated_text = get_prob_with_sentence(processor, model, prompt_1, image, device=device, model_mod=model_mod)
        yes_probs.append(yes_prob_1)
        no_probs.append(no_prob_1)

        # Generation and compute prob for sub
        yes_prob_2, no_prob_2, generated_text = get_prob_with_sentence(processor, model, prompt_2, image, device=device, model_mod=model_mod)
        yes_probs.append(yes_prob_2)
        no_probs.append(no_prob_2)

        # Generate additional context if both entities exist
        if yes_probs[0] > no_probs[0] and yes_probs[1] > no_probs[1]:
            if select_best_one:
                # Check if current image has the highest yes probability
                total_yes_prob = sum(yes_probs)
                if total_yes_prob > best_yes_prob:
                    best_yes_prob = total_yes_prob
                    best_no_prob = sum(no_probs)
                    best_image = image
            else:
                useful_images.append(image)
                print(f"\t\t>>>> Image: {image_path} | Yes: {[round(yes_prob, 6) for yes_prob in yes_probs]}, No: {[round(no_prob, 6) for no_prob in no_probs]}")

    # Add the best image to useful_images if select_best_one is True
    if select_best_one and best_image is not None:
        useful_images.append(best_image)
        print("\t\t>>>> Best Image: {image_path} | Yes: {round(best_yes_prob, 6)}, Yes: {round(best_no_prob, 6)}")

    if len(useful_images) == 0:
        print("\t\t>>>> No useful images!")

    return useful_images

# Summarize given set of images
def get_summary_context(model, processor, useful_images, summary_prompt):
    context_list = []

    inputs = processor(images=useful_images, text=summary_prompt, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    inputs['img_mask'] = torch.tensor([[1 for i in range(len(useful_images))]])
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

    inputs = inputs.to('cuda:0')
    with torch.no_grad():
        context_outputs = model.generate(
            pixel_values=inputs['pixel_values'],
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            img_mask=inputs['img_mask'],
            do_sample=False,
            max_length=50,
            min_length=1,
            set_min_padding_size=False,
            sp_token=processor.tokenizer.img_place_token_id
        )

    context_text = processor.batch_decode(
        context_outputs, skip_special_tokens=True
    )[0].strip()

    context_list.append(context_text)
    print(f'\t>>>> Generated context: {context_text}')

    return context_list

# Setting_3
def get_ent_images(ent_path):
    # check if the path is valid\
    if not os.path.exists(ent_path):
        return None
    # collect all images
    ent_images = []
    for image_path in os.listdir(ent_path):
        yes_probs, no_probs = [], []
        image_full_path = os.path.join(ent_path, image_path)

        # print(f"Content of '{image_full_path}'")
        try:
            image = Image.open(image_full_path).convert("RGB")
        except Exception as e:
            print(f"Error during opening image {image_full_path}:", e)
            continue
        # resize image if its size is 1, 1
        if image.size == (1, 1) or image.size[0] == 1 or image.size[1] == 1:
            print("Resizing the image, original size is 1*n or n*1, will cause encoder issue...")
            image = image.resize((10, 10))
        ent_images.append(image)
    return ent_images

# generate summary for entity
def ent_images_summary(ent_images, ent_str, token_replace_str, model, processor, prompt_list, device, model_mod="single"):
    summary_prompt = prompt_list[0]
    if len(ent_images) > 25:
        ent_images = ent_images[:25]
    # remove the last ','
    token_replace_str = token_replace_str[:-1].strip()
    summary_prompt = summary_prompt.format(token_replace_str=token_replace_str, ent_str=ent_str)
    # print(f"entity: {ent_str}, prompt: {summary_prompt}")

    if model_mod == 'single':  # only use the first image if using single-image model
        context_inputs = processor(
            images=ent_images[0], text=summary_prompt, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            context_outputs = model.generate(
                **context_inputs,
                do_sample=True,  # generate with sample may return conflict with prob
                # num_beams=5,
                max_length=1024,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,  # generate with greedy or not
                output_scores=True,
                return_dict_in_generate=True
            )
    elif model_mod == 'multi':
        context_inputs = processor(images=ent_images, text=summary_prompt, return_tensors="pt")
        context_inputs['pixel_values'] = context_inputs['pixel_values'].to(torch.bfloat16)
        context_inputs['img_mask'] = torch.tensor([[1 for i in range(len(ent_images))]])
        context_inputs['pixel_values'] = context_inputs['pixel_values'].unsqueeze(0)

        context_inputs = context_inputs.to(device)
        with torch.no_grad():
            context_outputs = model.generate(
                pixel_values=context_inputs['pixel_values'],
                input_ids=context_inputs['input_ids'],
                attention_mask=context_inputs['attention_mask'],
                img_mask=context_inputs['img_mask'],
                do_sample=False,
                max_length=50,
                min_length=1,
                set_min_padding_size=False,
                sp_token=processor.tokenizer.img_place_token_id
            )

    context_text = processor.batch_decode(
        context_outputs, skip_special_tokens=True
    )[0].strip()

    # print(f'\t>>>> Generated context: {context_text}')
    return context_text

# setting_3 have two context: comonsense focus and image focus
def ent_combine_summary(ent_images, ent_str, replace_token, model, processor, device, model_mod="single"):
    if len(ent_images) == 0:
        # DEBUG
        print(f"No images provided for entity {ent_str}.")
        return ''

    # prompt format
    token_replace_str = " ".join(
        [
            f"image {idx}: <image{idx}>{replace_token},"
            for idx in range(len(ent_images))
        ]
    )

    ent_commonsense_summary_prompt = (
        "Analyze the set of images {token_replace_str}. "
        "Use the visual information from these images and commonsense knowledge from yourself to generate a complete, informative description about {ent_str}. "
        "In one or more full sentences, answer what/who is {ent_str} "
        "focus more on commonsense knowledge and combine some visually evidence about {ent_str}."
    ).format(token_replace_str=token_replace_str, ent_str=ent_str)
    ent_visual_summary_prompt = (
        "Analyze the set of images {token_replace_str}. "
        "In one or more full sentences, what can you tell about {ent_str} based on these images?"
        "Use only the visual information that can be clearly identified from these images; "
    ).format(token_replace_str=token_replace_str, ent_str=ent_str)

    # prompt_list = [ent_summary_prompt]

    ent_commonsense_summary_context = ent_images_summary(
        ent_images=ent_images, ent_str=ent_str,
        token_replace_str=token_replace_str,
        model=model, processor=processor,
        prompt_list=[ent_commonsense_summary_prompt],
        device=device, model_mod=model_mod
    )
    ent_visual_summary_context = ent_images_summary(
        ent_images=ent_images, ent_str=ent_str,
        token_replace_str=token_replace_str,
        model=model, processor=processor,
        prompt_list=[ent_visual_summary_prompt],
        device=device, model_mod=model_mod
    )

    # combine context sentences
    ent_idx = len(ent_str.split('_'))
    ent_format_s = ent_str.replace('_', ' ')
    ent_context_str = []
    if ent_commonsense_summary_context is not None:
        ent_commonsense_summary_context = ent_commonsense_summary_context.replace('_', ' ')
        if len(ent_commonsense_summary_context.split(' ')) <= ent_idx or f'{ent_format_s} is' not in ent_commonsense_summary_context:
            ent_commonsense_summary_context = f"{ent_str.replace('_',' ')} is {ent_commonsense_summary_context}"
        ent_context_str.append(ent_commonsense_summary_context)
    if ent_visual_summary_context is not None:
        ent_visual_summary_context = ent_visual_summary_context.replace('_', ' ')
        if len(ent_visual_summary_context.split(' ')) <= ent_idx or f'{ent_format_s} is' not in ent_visual_summary_context:
            ent_visual_summary_context = f"{ent_str.replace('_',' ')} is {ent_visual_summary_context}"
        ent_context_str.append(ent_visual_summary_context)

    ent_context_str = ' And '.join(ent_context_str)
    return ent_context_str
