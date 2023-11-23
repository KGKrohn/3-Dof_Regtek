def gui_hsv(hmin_v,hmax_v,smin_v,smax_v,vmin_v,vmax_v, hsv_queue):
    # ------------------------------------------------------------------------------------------------
    root = Tk()
    root.resizable(0, 0)
    # ------------------------------------------- Simple controller GUI definition-------------------------------------------
    w2 = Label(root, justify=LEFT, text="hsv")
    w2.config(font=("Elephant", 30))
    w2.grid(row=3, column=0, columnspan=2, padx=100)

    S1Lb = Label(root, text="hmin")
    S1Lb.config(font=("Elephant", 15))
    S1Lb.grid(row=5, column=0, pady=10)

    S2Lb = Label(root, text="hmax")
    S2Lb.config(font=("Elephant", 15))
    S2Lb.grid(row=10, column=0, pady=10)

    S3Lb = Label(root, text="smin")
    S3Lb.config(font=("Elephant", 15))
    S3Lb.grid(row=15, column=0, pady=10)

    S4Lb = Label(root, text="smax")
    S4Lb.config(font=("Elephant", 15))
    S4Lb.grid(row=20, column=0, pady=10)

    S5Lb = Label(root, text="vmin")
    S5Lb.config(font=("Elephant", 15))
    S5Lb.grid(row=25, column=0, pady=10)

    S6Lb = Label(root, text="vmax")
    S6Lb.config(font=("Elephant", 15))
    S6Lb.grid(row=30, column=0, pady=10)

    hmin = Scale(root, from_=0, to=255, orient=HORIZONTAL,
                 resolution=1, length=400, command=hmin_v)
    hmin.set(hmin_v)
    hmin.grid(row=5, column=1)

    hmax = Scale(root, from_=0, to=255, orient=HORIZONTAL,
                 resolution=1, length=400, command=hmax_v)
    hmax.grid(row=10, column=1)
    hmax.set(hmax_v)

    smin = Scale(root, from_=0, to=255, orient=HORIZONTAL,
                 resolution=1, length=400, command=smin_v,)
    smin.grid(row=15, column=1)
    smin.set(smin_v)

    smax = Scale(root, from_=0, to=255, orient=HORIZONTAL,
                 resolution=1, length=400, command=smax_v)
    smax.grid(row=20, column=1)
    smax.set(smax_v)

    vmin = Scale(root, from_=0, to=255, orient=HORIZONTAL,
                 resolution=1, length=400, command=vmin_v)
    vmin.grid(row=25, column=1)
    vmin.set(vmin_v)

    vmax = Scale(root, from_=0, to=255, orient=HORIZONTAL,
                resolution=1, length=400, command=vmax_v)
    vmax.grid(row=30, column=1)
    vmax.set(vmax_v)

    hsv_queue.put([hmin_v, hmax_v, smin_v, smax_v, vmin_v, vmax_v])

    root.mainloop()  # running loop



    hsv_queue = Queue()
    p3 = mp.Process(target=gui_hsv, args=(hmin_v, hmax_v, smin_v, smax_v, vmin_v, vmax_v, hsv_queue))  # initiate GUI controls