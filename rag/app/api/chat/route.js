// Step 1: Set up the file and imports
import { NextResponse } from 'next/server'
import { Pinecone } from '@pinecone-database/pinecone'
import { TextServiceClient } from '@google-cloud/ai-generative'

// Step 2: Define the system prompt
const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
Use them to answer the question if needed.
`

// Step 3: Create the POST function
export async function POST(req) {
    const data = await req.json()

    // Step 4: Initialize Pinecone and Google Generative AI
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const client = new TextServiceClient({
        keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS,
    })

    // Step 5: Process the user’s query
    const text = data[data.length - 1].content

    // Step 6: Query Pinecone
    const embedding = await client.embedText({
        input: text,
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.embeddings[0].values,
    })

    // Step 7: Format the results
    let resultString = ''
    results.matches.forEach((match) => {
        resultString += `
    Returned Results:
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`
    })

    // Step 8: Prepare the Google Generative AI request
    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    // Step 9: Send request to Google Generative AI
    const [completion] = await client.generateMessage({
        model: 'chat-bison',
        prompt: {
            messages: [
                { role: 'system', content: systemPrompt },
                ...lastDataWithoutLastMessage,
                { role: 'user', content: lastMessageContent },
            ],
        },
    })

    // Step 10: Set up streaming response
    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for (const chunk of completion.candidates) {
                    const content = chunk.message?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                }
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        },
    })

    // Return the streaming response
    return new NextResponse(stream)
}
