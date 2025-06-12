document.addEventListener('DOMContentLoaded', function() {
    const generateQuestionsBtn = document.getElementById('generateQuestionsBtn');
    const testLLMBtn = document.getElementById('testLLMBtn');
    const searchBasedBtn = document.getElementById('searchBasedBtn');
    const calculateSimilarityBtn = document.getElementById('calculateSimilarityBtn');
    const deepAnalysisBtn = document.getElementById('deepAnalysisBtn');
    const originalContent = document.getElementById('originalContent');
    const promptInput = document.getElementById('promptInput');
    const modelSelect = document.getElementById('modelSelect');
    const embeddingModel = document.getElementById('embeddingModel');
    const questionsContainer = document.getElementById('questionsContainer');
    const loader = document.getElementById('loader');
    const resultCard = document.getElementById('resultCard');
    const searchBasedSection = document.getElementById('searchBasedSection');
    const similarityScoresSection = document.getElementById('similarityScoresSection');
    const llmResponseEditable = document.getElementById('llmResponseEditable');
    const originalContentDisplay = document.getElementById('originalContentDisplay');
    const scrapedContent = document.getElementById('scrapedContent');
    const searchBasedAnswer = document.getElementById('searchBasedAnswer');
    const originalVsLLM = document.getElementById('originalVsLLM');
    const originalVsSearch = document.getElementById('originalVsSearch');
    const llmVsSearch = document.getElementById('llmVsSearch');
    const deepAnalysisSection = document.getElementById('deepAnalysisSection');
    const deepAnalysisResult = document.getElementById('deepAnalysisResult');
    
    let savedOriginalContent = "";
    let savedLLMResponse = "";
    let savedSearchBasedAnswer = "";

    generateQuestionsBtn.addEventListener('click', async function() {
        const content = originalContent.value.trim();
        savedOriginalContent = content;
        loader.style.cssText = "display: flex !important; justify-content: center;";
        // Clear existing questions
        const questionsList = document.getElementById('questionsList');
        const dropdownBtn = document.getElementById('questionDropdownBtn');
        questionsList.innerHTML = '';
        
        try {
            const response = await fetch('http://localhost:8081/generate-questions', {
                method: 'POST',headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    text: content
                })
            });
            if (!response.ok) {
                throw new Error('Failed to generate questions');
            }
            const data = await response.json();
            data.questions.forEach((question, index) => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.className = 'dropdown-item text-truncate';
                a.href = '#';
                
                const displayText = question.length > 80 ? question.substring(0, 80) + '...' : question;
                a.textContent = displayText;
                a.title = question;
                
                a.addEventListener('click', function(e) {
                    e.preventDefault();
                    promptInput.value = question;
                    dropdownBtn.textContent = displayText;
                    dropdownBtn.title = question; // Add title for hover on button too
                });
                
                li.appendChild(a);
                questionsList.appendChild(li);
                if (index === 0) {
                    promptInput.value = question;
                    dropdownBtn.textContent = displayText;
                    dropdownBtn.title = question;
                }
            });
            questionsContainer.classList.remove('d-none');

        } catch (error) {
            console.error('Error:', error);
            alert('Error generating questions: ' + error.message);
        } finally {
            loader.style.cssText = "display: none !important;";
        }
    });

    testLLMBtn.addEventListener('click', async function() {
        loader.style.cssText = "display: flex !important; justify-content: center;";
        resultCard.classList.add('d-none');
        deepAnalysisSection.classList.add('d-none');
        searchBasedSection.classList.add('d-none');
        similarityScoresSection.classList.add('d-none');
        
        try {
            // Get LLM response
            const responseData = await fetch('http://localhost:8081/test-llm', {
                method: 'POST',headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    prompt: promptInput.value.trim(),
                    model_id: modelSelect.value,
                    embedding_model: embeddingModel.value,
                })
            });

            if (!responseData.ok) {
                throw new Error('Failed to get LLM response');
            }
            
            const data = await responseData.json();
            savedLLMResponse = data.llm_response;
            llmResponseEditable.value = data.llm_response;
            originalContentDisplay.innerHTML = marked.parse(savedOriginalContent);
            resultCard.classList.remove('d-none');
            deepAnalysisBtn.disabled = false;
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error testing LLM: ' + error.message);
        } finally {
            loader.style.cssText = "display: none !important;";
        }
    });

    searchBasedBtn.addEventListener('click', async function() {
        loader.style.cssText = "display: flex !important; justify-content: center;";
        similarityScoresSection.classList.add('d-none');
        
        try {
            const response = await fetch('http://localhost:8081/search-based-answer', {
                method: 'POST',headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    prompt: promptInput.value.trim(),
                    original_content: savedOriginalContent
                })
            });
            
            const data = await response.json();
            savedSearchBasedAnswer = data.search_based_answer;
            searchBasedAnswer.value = data.search_based_answer;
            scrapedContent.innerHTML = marked.parse(data.scraped_content);
            searchBasedSection.classList.remove('d-none');
            resultCard.classList.remove('d-none');
            
        } catch (error) {
            console.error('Error:', error);
            searchBasedSection.classList.remove('d-none');
            resultCard.classList.remove('d-none');
        } finally {
            loader.style.cssText = "display: none !important;";
        }
    });

    calculateSimilarityBtn.addEventListener('click', async function() {
        savedLLMResponse = llmResponseEditable.value;
        savedSearchBasedAnswer = searchBasedAnswer.value;
        loader.style.cssText = "display: flex !important; justify-content: center;";
        
        try {
            const response = await fetch('http://localhost:8081/calculate-similarity', {
                method: 'POST',headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    original_content: savedOriginalContent,
                    llm_response: savedLLMResponse,
                    search_based_answer: savedSearchBasedAnswer
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to calculate similarity scores');
            }
            const data = await response.json();
            
            const originalVsLLMScore = Math.round(data.original_vs_llm * 100);
            const originalVsSearchScore = Math.round(data.original_vs_search * 100);
            const llmVsSearchScore = Math.round(data.llm_vs_search * 100);
            
            originalVsLLM.textContent = `${originalVsLLMScore}%`;
            originalVsLLM.style.width = `${originalVsLLMScore}%`;
            
            originalVsSearch.textContent = `${originalVsSearchScore}%`;
            originalVsSearch.style.width = `${originalVsSearchScore}%`;
            
            llmVsSearch.textContent = `${llmVsSearchScore}%`;
            llmVsSearch.style.width = `${llmVsSearchScore}%`;
            similarityScoresSection.classList.remove('d-none');
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error calculating similarity scores: ' + error.message);
        } finally {
            loader.style.cssText = "display: none !important;";
        }
    });

    deepAnalysisBtn.addEventListener('click', async function() {
        savedLLMResponse = llmResponseEditable.value;
        deepAnalysisSection.classList.remove('d-none');
        deepAnalysisResult.textContent = "Analyzing...";
        loader.style.cssText = "display: flex !important; justify-content: center;";
        
        try {
            const response = await fetch('http://localhost:8081/analyze-response', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    original_content: savedOriginalContent,
                    llm_response: savedLLMResponse
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to get expert analysis');
            }
            
            const data = await response.json();
            deepAnalysisResult.innerHTML = marked.parse(data.analysis);
            
        } catch (error) {
            console.error('Error:', error);
            deepAnalysisResult.textContent = 'Error getting expert analysis: ' + error.message;
        } finally {
            loader.style.cssText = "display: none !important;";
        }
    });
});