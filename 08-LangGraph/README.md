# LangGraph 1: Conditional customer balance flow

Run from the existing project environment:

```powershell
F:\Work\AI-Engineer-Bootcamp\.venv\Scripts\python.exe F:\Work\AI-Engineer-Bootcamp\08-LangGraph\conditional_customer_flow.py
```

The graph follows `START → FindCustomer → GetBalance → conditional route`. A balance below ₹10,000 visits `SendEmail` and then `END`; otherwise the route goes straight to `END`. `SendEmail` prints a simulated message and does not send mail.

| LangGraph concept | In this exercise |
| --- | --- |
| State | `CustomerState`, the typed values passed between nodes |
| Nodes | `FindCustomer`, `GetBalance`, and `SendEmail` functions |
| Edges | Fixed transitions from `START` through lookup and balance |
| Conditional edge | `choose_next` selects the low-balance node or `END` |
| Compile and invoke | `compile()` builds the graph; `invoke()` runs each sample |

The two built-in sample customers cover both paths, and assertions check the visited nodes.
