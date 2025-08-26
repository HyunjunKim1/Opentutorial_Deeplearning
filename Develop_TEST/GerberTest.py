import os

def process_gerber(filename: str):
    ret: list[float] = []

    if filename == "":
        return None

    # 경로 문자열 처리
    filename = filename.replace("\\\\", "\\")

    if not os.path.exists(filename):
        return None

    val = GerberValidator()
    if not val.do_validate(filename, GerberLayerType.Unit):
        return None

    m_ptrGerberLayer = generate_gerber_layer(GerberLayerType.Unit, False, True, filename)

    if len(m_ptrGerberLayer.section_data) > 0:
        ret.append(0.0)
        for i in range(len(m_ptrGerberLayer.section_data) - 1):
            diff = (m_ptrGerberLayer.section_data[0].section_top -
                    m_ptrGerberLayer.section_data[i + 1].section_top)
            ret.append(diff)

    # C# RemoveRange → Python del 슬라이싱
    if len(ret) == 8:
        del ret[1:2]  # index 1
        del ret[2:3]  # index 2 (기존 리스트에서 1개 삭제됐으므로 주의)
        del ret[3:4]
        del ret[4:5]

    if len(ret) == 12:
        del ret[1:3]  # index 1~2
        del ret[2:4]  # index 2~3
        del ret[3:5]  # index 3~4
        del ret[4:6]  # index 4~5

    return ret
