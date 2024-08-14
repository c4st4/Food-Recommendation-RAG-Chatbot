import streamlit as st
import pandas as pd
import google.generativeai as genai
import uuid

# Configure Gemini (make sure to use st.secrets for the API key in production)
genai.configure(api_key="AIzaSyAmK6R8MhwYdHCESNDmcP2IwypCHx_EJeI")
model = genai.GenerativeModel('gemini-1.5-flash')

# Load your data without caching
def load_data():
    return pd.read_csv('data1.csv')

df = load_data()

def generate_query(prompt, sample_data, columns, context):
    query_prompt = f"""
    Given the following prompt, sample data, column names from a DataFrame, and previous context, generate a pandas DataFrame operation to answer the question:
    Rating of restaurant is in decimals, like 4.5, 4.6, 5.0. Give all the columns of the dataset in the result.
    The number of reviews of a restaurant are given under the column number of reviews.
    Make sure the pandas operation only returns 400 rows at max.
    The tags column are what main items and cusine the restaurant serves.
    All restaurants are in the Islamabad area.
    If the prompt has an specific area mentioned for example F-11 then also search for the area without the hyphen e.g 'F11' or 'F 11', the sector is given in the addresses.
    If an address is given in the prompt only give restaurants in that area.
    For food items or cuisines, search in the 'tags' and 'name' column using partial string matching.
    If the prompt mentions 'good' or 'best', consider restaurants with ratings above 4.5.
    You can also be asked food that is within a specific price range you can see the price in the 'price' column.
    When asked about 'Desserts' also search for ice cream and cakes in 'name' and 'tags' the pandas operation.
    
    
    Sample data (5 random rows):
    {sample_data}
    Available columns:
    {columns}
    Previous context:
    {context}
    Prompt: {prompt}
    Respond with a pandas operation that will answer the question.
    Only provide the pandas code, without any additional explanation or formatting.
    Use standard pandas methods like .loc[], .iloc[], or boolean indexing.
    Do not use methods like .query() or custom functions.
    The operation should be a single line of pandas code.
    Example formats:
    df[df['column_name'] == value]
    df.loc[df['column_name'] > value, ['column1', 'column2']]
    """
    response = model.generate_content(query_prompt)
    st.write(response.text.strip())
    return response.text.strip()

def execute_query(query):
    query = query.replace('```python', '').replace('```', '').strip()
    if query.startswith('df[') or query.startswith('df.'):
        query = query[2:]
    try:
        result = eval(f"df{query}")
        if isinstance(result, pd.Series):
            result = result.to_frame()
        elif not isinstance(result, pd.DataFrame):
            result = pd.DataFrame({'Result': [result]})
        return result
    except Exception as e:
        raise ValueError(f"Error executing pandas operation: {str(e)}")

def query_to_context(df_result):
    return df_result.to_string(index=False)
def generate_response(prompt, context, chat_history):
    response_prompt = f"""
    Given the following data, prompt, and chat history, provide a comprehensive and detailed answer:
    Data:
    {context}
    Prompt: {prompt}
    Chat History:
    {chat_history}
    Please provide a short analysis of the data in relation to the prompt and chat history. Include specific examples from the data, explain any trends or patterns you notice, and offer insights that would be valuable to someone interested in this information. Your response should be well-structured and informative
    Do not mention any downfalls or limitations.
    If the prompt is not related to restaurants, respond with: 'This query is not related to restaurants.'.
    If the pandas operation returned an empty DataFrame, respond with: "I couldn't find any restaurants that match the criteria."
    Provide the address for all the restaurants in the result.

    Response:
    """
    response = model.generate_content(response_prompt)
    return response.text.strip()

def rag_with_data_query(prompt, chat_id):
    try:
        random_sample = df.sample(n=5, random_state=42)
        sample_data = random_sample.to_string()
        columns = ', '.join(df.columns)
        
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chats[chat_id]['history']])
        
        query = generate_query(prompt, sample_data, columns, context)
        result_df = execute_query(query)
        new_context = query_to_context(result_df)
        response = generate_response(prompt, new_context, context)
        
        return query, result_df, response
    except Exception as e:
        return str(e), str(e), str(e)


# Define suggested questions
suggested_questions = [
    "What are some good Pakistani restaurants in F-10?",
    "Which are the best restaurants for beef burgers in f11?"
]

# Initialize session state for storing chats
if 'chats' not in st.session_state:
    st.session_state.chats = {}

if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None

# Streamlit UI
st.title("Restaurant Recommendation RAG Model")

# Sidebar for chat management
with st.sidebar:
    st.subheader("Chats")
    if st.button("New Chat"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {"name": f"Chat {len(st.session_state.chats) + 1}", "history": []}
        st.session_state.current_chat_id = new_chat_id

    for chat_id, chat_data in st.session_state.chats.items():
        if st.button(chat_data["name"], key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id

 #   st.subheader("Dataset Information")
  #  st.write(f"Total restaurants: {len(df)}")
  #  st.write(f"Columns: {', '.join(df.columns)}")

# Main chat area
if st.session_state.current_chat_id:
    st.subheader(f"Current Chat: {st.session_state.chats[st.session_state.current_chat_id]['name']}")
    
    # Display chat history
    for q, a in st.session_state.chats[st.session_state.current_chat_id]['history']:
        st.text_area("Question:", value=q, height=50, disabled=True)
        st.markdown(a)
        st.markdown("---")

    # Add suggested questions section
    st.subheader("Suggested Questions")
    for question in suggested_questions:
        if st.button(question):
            user_prompt = question
            with st.spinner("Generating recommendations..."):
                query, result_df, response = rag_with_data_query(user_prompt, st.session_state.current_chat_id)
          #      print(query)
                # Add the new query and results to the chat history
                st.session_state.chats[st.session_state.current_chat_id]['history'].append((user_prompt, response))
                
                # Display the latest result
                st.subheader("Latest Result")
                st.write(f"Prompt: {user_prompt}")
              #  st.subheader("Generated Query")
              #  st.code(query, language="python")
                
             #   if result_df is not None:
             #       st.subheader("Query Result")
             #       st.dataframe(result_df)
                
                st.subheader("Final Response")
                st.write(query)
                st.markdown(response)

            # Force a rerun to update the chat history display
        #    st.experimental_rerun()

    # Create a form for the input and button
    with st.form(key='query_form'):
        user_prompt = st.text_input("Enter your query about restaurants:")
        submit_button = st.form_submit_button("Get Recommendations")


    if submit_button and user_prompt:
        with st.spinner("Generating     recommendations..."):
            query, result_df, response = rag_with_data_query(user_prompt, st.session_state.current_chat_id)
            
            # Add the new query and results to the chat history
            st.session_state.chats[st.session_state.current_chat_id]['history'].append((user_prompt, response))
            
            # Display the latest result
            st.subheader("Latest Result")
            st.write(f"Prompt: {user_prompt}")
          #  st.subheader("Generated Query")
           # st.code(query, language="python")
            
 #           if result_df is not None:
            #    st.subheader("Query Result")
             #   st.dataframe(result_df)
            
            st.subheader("Final Response")
            st.markdown(response)

        # Force a rerun to update the chat history display
    #    st.experimental_rerun()
else:
    st.info("Please create a new chat or select an existing one from the sidebar.")