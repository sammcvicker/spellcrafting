# Magically Abstract

Magically is LLMs as a Python language feature.

It's powered by PydanticAI - an omni-provider agentic SDK.

It inspects functions with the `@spell` decorator for prompt and output type.

Codepaths simply call spells like regular functions, they return structured output. Model/tool overrides can be given in decorator params.

Configuration is done all in pyproject.toml for simple stuff. Can support runtime config for more production stuff.
